import { useCallback, useEffect, useRef, useState } from "react";
import {
	applyInteractiveCommit,
	clearInteractivePreviewSegmentation,
	createInteractivePreviewSegmentation,
	setInteractivePreviewRegion,
	worldToInteractiveIJK,
} from "../CornerstoneNifti2";
import { sliceBounds, sliceForPlane, type IJK, type ViewerPlane } from "../interactiveSegmentation";
import { API_BASE } from "../constants";
import type { LiveRoomController } from "../../liveRooms/types";

type Region = {
	bbox: [[number, number], [number, number], [number, number]];
	shape: [number, number, number];
	encoding: "zlib-base64-uint8";
	data: string;
};

async function interactiveRequest<T>(path: string, body?: Record<string, unknown>, roomKey?: string): Promise<T> {
	const response = await fetch(`${API_BASE}/api/interactive/${path}`, {
		method: body ? "POST" : "GET",
		credentials: "include",
		headers: body ? { "Content-Type": "application/json", ...(roomKey ? { "X-Room-Key": roomKey } : {}) } : undefined,
		body: body ? JSON.stringify(body) : undefined,
	});
	const data = await response.json().catch(() => ({}));
	if (!response.ok) {
		const error = new Error(typeof data.error === "string" ? data.error : `Interactive request failed (${response.status})`);
		(error as Error & { status?: number; code?: string }).status = response.status;
		(error as Error & { status?: number; code?: string }).code = data.code;
		throw error;
	}
	return data as T;
}

async function decodeRegion(region: Region): Promise<Uint8Array> {
	const binary = atob(region.data);
	const compressed = Uint8Array.from(binary, (char) => char.charCodeAt(0));
	const stream = new Blob([compressed]).stream().pipeThrough(new DecompressionStream("deflate"));
	const bytes = new Uint8Array(await new Response(stream).arrayBuffer());
	const expected = region.shape.reduce((product, value) => product * value, 1);
	if (bytes.length !== expected) throw new Error("The preview mask could not be decoded");
	return bytes;
}

function buildSliceCrop(points: IJK[], axis: 0 | 1 | 2, sliceIndex: number, lasso: boolean) {
	const planeAxes = ([0, 1, 2] as const).filter((value) => value !== axis);
	const starts = [0, 0, 0];
	const ends = [0, 0, 0];
	starts[axis] = sliceIndex;
	ends[axis] = sliceIndex + 1;
	for (const currentAxis of planeAxes) {
		starts[currentAxis] = Math.min(...points.map((point) => point[currentAxis]));
		ends[currentAxis] = Math.max(...points.map((point) => point[currentAxis])) + 1;
	}
	const shape = ends.map((end, i) => end - starts[i]) as [number, number, number];
	if (shape.reduce((size, next) => size * next, 1) > 1_000_000) throw new Error("Prompt crop is too large; draw a smaller mark");
	const values = new Uint8Array(shape[0] * shape[1] * shape[2]);
	const local = points.map((point) => [point[planeAxes[0]] - starts[planeAxes[0]], point[planeAxes[1]] - starts[planeAxes[1]]]);
	const set = (x: number, y: number) => {
		const ijk = [0, 0, 0];
		ijk[axis] = 0;
		ijk[planeAxes[0]] = x;
		ijk[planeAxes[1]] = y;
		values[(ijk[0] * shape[1] + ijk[1]) * shape[2] + ijk[2]] = 1;
	};
	if (lasso && local.length >= 3) {
		for (let y = 0; y < shape[planeAxes[1]]; y++) for (let x = 0; x < shape[planeAxes[0]]; x++) {
			let inside = false;
			for (let i = 0, j = local.length - 1; i < local.length; j = i++) {
				const [xi, yi] = local[i]; const [xj, yj] = local[j];
				if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) inside = !inside;
			}
			if (inside) set(x, y);
		}
	} else {
		const radius = 1;
		for (let p = 0; p < local.length; p++) {
			const from = local[Math.max(0, p - 1)]; const to = local[p];
			const steps = Math.max(1, Math.ceil(Math.max(Math.abs(to[0] - from[0]), Math.abs(to[1] - from[1])) * 2));
			for (let n = 0; n <= steps; n++) {
				const x = Math.round(from[0] + (to[0] - from[0]) * n / steps);
				const y = Math.round(from[1] + (to[1] - from[1]) * n / steps);
				for (let dy = -radius; dy <= radius; dy++) for (let dx = -radius; dx <= radius; dx++) {
					if (x + dx >= 0 && y + dy >= 0 && x + dx < shape[planeAxes[0]] && y + dy < shape[planeAxes[1]]) set(x + dx, y + dy);
				}
			}
		}
	}
	const crop = Array.from({ length: shape[0] }, (_, i) => Array.from({ length: shape[1] }, (_, j) =>
		Array.from({ length: shape[2] }, (_, k) => values[(i * shape[1] + j) * shape[2] + k])));
	return { bbox: starts.map((start, i) => [start, ends[i]]) as Region["bbox"], crop };
}

export type InteractiveStatus = "disabled" | "loading" | "idle" | "starting" | "ready" | "predicting" | "error";

export function useInteractiveSegmentation(
	caseId: string | number | null,
	isCvCase = false,
	fullResolutionReady = false,
	room?: Pick<LiveRoomController, "metadata" | "roomKey" | "pendingEvents" | "sendDurable">,
	selectedLabelId: number | null = null,
	selectedLabelName = "",
) {
	const [enabled, setEnabled] = useState(false);
	const [status, setStatus] = useState<InteractiveStatus>("loading");
	const [message, setMessage] = useState("");
	const [sessionId, setSessionId] = useState<string | null>(null);
	const sessionRef = useRef<string | null>(null);
	const generationRef = useRef(0);
	const roomPromptEventsRef = useRef(new Set<string>());
	const localAcceptedEventsRef = useRef(new Set<string>());
	const roomChunksRef = useRef(new Map<string, { expected: number; values: string[]; meta: Record<string, any> }>());
	const roomApplyQueueRef = useRef<Promise<void>>(Promise.resolve());
	const roomPromptTargetRef = useRef<{ id: number; name: string } | null>(null);
	const selectedLabelRef = useRef<number | null>(selectedLabelId);
	const organStatsRef = useRef<Record<number, { label: string; promptCount: number; prompts: string[] }>>({});
	const [organStatus, setOrganStatus] = useState<Record<number, { label: string; promptCount: number; prompts: string[] }>>({});

	useEffect(() => {
		let active = true;
		void interactiveRequest<{ enabled: boolean }>("config").then((config) => {
			if (!active) return;
			setEnabled(config.enabled);
			setStatus(config.enabled ? "idle" : "disabled");
		}).catch(() => {
			if (active) { setEnabled(false); setStatus("disabled"); }
		});
		return () => { active = false; };
	}, []);

	useEffect(() => {
		organStatsRef.current = {};
		setOrganStatus({});
		roomPromptTargetRef.current = null;
		roomPromptEventsRef.current.clear();
		roomChunksRef.current.clear();
		localAcceptedEventsRef.current.clear();
	}, [caseId, room?.metadata.room_id]);

	const close = useCallback(async () => {
		const current = sessionRef.current;
		sessionRef.current = null;
		setSessionId(null);
		clearInteractivePreviewSegmentation();
		if (current) {
			try { await interactiveRequest("close", { session_id: current, ...(room ? { room_id: room.metadata.room_id } : {}) }, room?.roomKey); } catch { /* idle reaper owns failed close */ }
		}
	}, [room?.metadata.room_id, room?.roomKey]);

	useEffect(() => {
		generationRef.current += 1;
		const current = sessionRef.current;
		sessionRef.current = null;
		setSessionId(null);
		clearInteractivePreviewSegmentation();
		if (current) void interactiveRequest("close", { session_id: current, ...(room ? { room_id: room.metadata.room_id } : {}) }, room?.roomKey).catch(() => {});
		return () => { generationRef.current += 1; };
	}, [caseId, room?.metadata.room_id, room?.roomKey]);

	const start = useCallback(async () => {
		if (sessionRef.current) return sessionRef.current;
		if (!enabled) throw new Error("Interactive segmentation is disabled on this server.");
		if (!fullResolutionReady) throw new Error("Load the full-resolution volume with the HD button before starting interactive segmentation.");
		if (caseId == null || isCvCase && String(caseId).toUpperCase().startsWith("LOCAL")) {
			throw new Error("Open a PanTS or CancerVerse dataset case to use interactive segmentation.");
		}
		if (selectedLabelId == null) throw new Error("Select an organ from the label list before starting interactive segmentation.");
		setStatus("starting");
		setMessage("Starting a research session…");
		const generation = generationRef.current;
		try {
			const data = await interactiveRequest<{ session_id: string; shape: number[]; seeded_mask: Region | null }>("start", {
				dataset: isCvCase ? "CancerVerse" : "PanTS",
				case_id: String(caseId),
				initial_mask_label: selectedLabelId,
				...(room ? { room_id: room.metadata.room_id } : {}),
			}, room?.roomKey);
			if (generation !== generationRef.current) {
				void interactiveRequest("close", { session_id: data.session_id }).catch(() => {});
				throw new Error("Case changed while the session was starting");
			}
			if (!await createInteractivePreviewSegmentation()) {
				void interactiveRequest("close", { session_id: data.session_id }).catch(() => {});
				throw new Error("Could not create the preview layer. Finish loading the full-resolution volume first.");
			}
			if (data.seeded_mask) setInteractivePreviewRegion(data.seeded_mask.bbox, data.seeded_mask.shape, await decodeRegion(data.seeded_mask));
			sessionRef.current = data.session_id;
			selectedLabelRef.current = selectedLabelId;
			setSessionId(data.session_id);
			setStatus("ready");
			setMessage(`Ready · ${data.shape.join(" × ")} voxels · research use only`);
			return data.session_id;
		} catch (error) {
			setStatus("error");
			setMessage(error instanceof Error ? error.message : "Could not start an interactive session");
			throw error;
		}
	}, [caseId, enabled, isCvCase, fullResolutionReady, selectedLabelId, room?.metadata.room_id, room?.roomKey]);

	useEffect(() => {
		if (!sessionRef.current || selectedLabelId == null || selectedLabelRef.current === selectedLabelId) return;
		let active = true;
		void interactiveRequest<{ seeded_mask: Region | null }>("select-organ", {
			session_id: sessionRef.current, label_id: selectedLabelId,
			...(room ? { room_id: room.metadata.room_id } : {}),
		}, room?.roomKey).then(async (result) => {
			if (!active) return;
			selectedLabelRef.current = selectedLabelId;
			clearInteractivePreviewSegmentation();
			await createInteractivePreviewSegmentation();
			if (result.seeded_mask) setInteractivePreviewRegion(result.seeded_mask.bbox, result.seeded_mask.shape, await decodeRegion(result.seeded_mask));
			setMessage(`Editing ${selectedLabelName || `label ${selectedLabelId}`}. Existing voxels loaded as the starting mask.`);
		}).catch((error) => { if (active) { setStatus("error"); setMessage(error instanceof Error ? error.message : "Could not switch the selected organ"); } });
		return () => { active = false; };
	}, [selectedLabelId, selectedLabelName, room?.metadata.room_id, room?.roomKey]);

	const submit = useCallback(async (
		plane: ViewerPlane,
		pointWorld: readonly number[],
		boxWorld?: readonly [readonly number[], readonly number[]],
		include = true,
		promptType: "point" | "box" | "scribble" | "lasso" = boxWorld ? "box" : "point",
		strokeWorld?: readonly (readonly number[])[],
	) => {
		const id = await start();
		const axis = sliceForPlane(plane);
		const point = worldToInteractiveIJK(pointWorld as [number, number, number]);
		const sliceIndex = point[axis];
			const base = { session_id: id, slice_axis: axis, slice_index: sliceIndex, include,
			...(room ? { room_id: room.metadata.room_id } : {}) };
		setStatus("predicting");
		if (room && selectedLabelId != null) roomPromptTargetRef.current = { id: selectedLabelId, name: selectedLabelName };
		setMessage("Predicting…");
		try {
			const result = (promptType === "scribble" || promptType === "lasso")
				? await interactiveRequest<{ delta: Region | null }>(promptType, (() => {
					const stroke = (strokeWorld ?? [pointWorld]).map((world) => worldToInteractiveIJK(world as [number, number, number]));
					const { bbox, crop } = buildSliceCrop(stroke, axis, sliceIndex, promptType === "lasso");
					return { ...base, interaction_bbox: bbox, crop };
				})(), room?.roomKey)
				: boxWorld
				? await interactiveRequest<{ delta: Region | null }>("bbox", {
					...base,
					bounds: sliceBounds(axis, sliceIndex, point, worldToInteractiveIJK(boxWorld[1] as [number, number, number])),
				}, room?.roomKey)
				: await interactiveRequest<{ delta: Region | null }>("point", { ...base, coordinates: point as IJK }, room?.roomKey);
			if (selectedLabelId != null) {
				const previous = organStatsRef.current[selectedLabelId] ?? { label: selectedLabelName || `Label ${selectedLabelId}`, promptCount: 0, prompts: [] };
				const effectivePrompt = boxWorld ? "box" : promptType;
				const promptText = `${effectivePrompt}${include ? " (+)" : " (-)"}`;
				const next = { ...previous, label: selectedLabelName || previous.label, promptCount: previous.promptCount + 1, prompts: [...previous.prompts, promptText] };
				organStatsRef.current = { ...organStatsRef.current, [selectedLabelId]: next };
				setOrganStatus(organStatsRef.current);
			}
			if (result.delta) {
				const bytes = await decodeRegion(result.delta);
				setInteractivePreviewRegion(result.delta.bbox, result.delta.shape, bytes);
				if (room) {
					const promptId = crypto.randomUUID();
					const compressed = result.delta.data;
					const chunkSize = 300_000;
					const chunkCount = Math.ceil(compressed.length / chunkSize);
					const prompt = promptType === "point" ? { type: promptType, coordinates: point }
						: boxWorld ? { type: "bbox", bounds: sliceBounds(axis, sliceIndex, point, worldToInteractiveIJK(boxWorld[1] as [number, number, number])) }
						: (() => {
							const points = (strokeWorld ?? [pointWorld]).map((world) => worldToInteractiveIJK(world as [number, number, number]));
							const crop = buildSliceCrop(points, axis, sliceIndex, promptType === "lasso");
							return { type: promptType, interaction_bbox: crop.bbox, points };
						})();
					for (let chunkIndex = 0; chunkIndex < chunkCount; chunkIndex++) {
						const committed = await room.sendDurable("interactive.prompt", {
							prompt_id: promptId, prompt_type: prompt.type, prompt: chunkIndex === 0 ? prompt : null,
							slice_axis: axis, slice_index: sliceIndex, include,
							bbox: result.delta.bbox, shape: result.delta.shape, encoding: result.delta.encoding,
							chunk_index: chunkIndex, chunk_count: chunkCount,
							data_chunk: compressed.slice(chunkIndex * chunkSize, (chunkIndex + 1) * chunkSize),
							label_id: selectedLabelId, organ_label: selectedLabelName,
							prompt_count: selectedLabelId == null ? 0 : (organStatsRef.current[selectedLabelId]?.promptCount ?? 0),
						}, `${promptId}:${chunkIndex}`);
						if (!committed) throw new Error("The Live Room did not save this prompt preview");
					}
				}
			}
			setStatus("ready");
			setMessage(result.delta ? "Preview updated. Accept to add it to the working labelmap." : "No mask change; try another prompt.");
			return result.delta ? 1 : 0;
		} catch (error) {
			setStatus("error");
			const statusCode = (error as Error & { status?: number }).status;
			setMessage(statusCode === 410 ? "The remote session expired. Existing prompts were restored; retry the action." :
				statusCode === 503 ? "The GPU server is full. Wait a moment and try again." :
				statusCode === 403 ? "A verified research account is required." :
				statusCode === 401 ? "Sign in to use interactive segmentation." :
				error instanceof Error ? error.message : "Interactive prediction failed.");
			throw error;
		}
	}, [room?.metadata.room_id, room?.roomKey, room?.sendDurable, start, selectedLabelId, selectedLabelName]);

	const undo = useCallback(async () => {
		if (!sessionRef.current) return;
		const result = await interactiveRequest<{ delta: Region | null }>("undo", { session_id: sessionRef.current, ...(room ? { room_id: room.metadata.room_id } : {}) }, room?.roomKey);
		if (result.delta) setInteractivePreviewRegion(result.delta.bbox, result.delta.shape, await decodeRegion(result.delta));
		setMessage(result.delta ? "Last prompt undone." : "There is no prompt to undo.");
	}, [room?.metadata.room_id, room?.roomKey]);

	const reset = useCallback(async () => {
		if (!sessionRef.current) return;
		const result = await interactiveRequest<{ seeded_mask: Region | null }>("reset", { session_id: sessionRef.current, ...(room ? { room_id: room.metadata.room_id } : {}) }, room?.roomKey);
		clearInteractivePreviewSegmentation();
		await createInteractivePreviewSegmentation();
		if (result.seeded_mask) setInteractivePreviewRegion(result.seeded_mask.bbox, result.seeded_mask.shape, await decodeRegion(result.seeded_mask));
		setMessage("Prompts cleared. The selected organ's starting mask is restored.");
	}, [room?.metadata.room_id, room?.roomKey]);

	const accept = useCallback(async (labelName: string, labelId: number, allowOverwrite = false) => {
		if (!sessionRef.current) throw new Error("Start an interactive session first");
		const acceptedLabelId = room ? (roomPromptTargetRef.current?.id ?? labelId) : labelId;
		const acceptedLabelName = room ? (roomPromptTargetRef.current?.name || labelName) : labelName;
		const result = await interactiveRequest<{ mask: Region; label_name: string; label_id: number }>("commit", {
			session_id: sessionRef.current, label_name: acceptedLabelName, label_id: acceptedLabelId,
			...(room ? { room_id: room.metadata.room_id } : {}),
		}, room?.roomKey);
		const bytes = await decodeRegion(result.mask);
		if (room) {
			const acceptId = crypto.randomUUID();
			localAcceptedEventsRef.current.add(acceptId);
			const compressed = result.mask.data;
			const chunkSize = 300_000;
			const chunkCount = Math.ceil(compressed.length / chunkSize);
			const stats = organStatsRef.current[acceptedLabelId];
			for (let chunkIndex = 0; chunkIndex < chunkCount; chunkIndex++) {
				const committed = await room.sendDurable("interactive.accept", {
					accept_id: acceptId, bbox: result.mask.bbox, shape: result.mask.shape,
					chunk_index: chunkIndex, chunk_count: chunkCount,
					data_chunk: compressed.slice(chunkIndex * chunkSize, (chunkIndex + 1) * chunkSize),
					label_id: result.label_id, organ_label: acceptedLabelName,
					prompt_count: stats?.promptCount ?? 0, prompt_log: stats?.prompts ?? [],
					allow_overwrite: allowOverwrite,
				}, `${acceptId}:${chunkIndex}`);
				if (!committed) {
					localAcceptedEventsRef.current.delete(acceptId);
					throw new Error("The Live Room did not save this accepted mask");
				}
			}
		}
		const changed = applyInteractiveCommit(result.mask.bbox, result.mask.shape, bytes, result.label_id, allowOverwrite);
		clearInteractivePreviewSegmentation();
		await createInteractivePreviewSegmentation();
		setMessage(`Accepted ${result.label_name} (${changed.toLocaleString()} voxels). Session remains open; select another organ to continue.`);
		return changed;
	}, [room?.metadata.room_id, room?.roomKey, room?.sendDurable]);

	useEffect(() => {
		if (!room?.pendingEvents.length || !room.metadata.room_id) return;
		for (const delivery of room.pendingEvents) {
			const event = delivery.event;
			if ((event.type !== "interactive.prompt" && event.type !== "interactive.accept") || roomPromptEventsRef.current.has(event.event_id)) continue;
			roomPromptEventsRef.current.add(event.event_id);
			const payload = event.payload as Record<string, any>;
			const isAccept = event.type === "interactive.accept";
			const promptId = String(isAccept ? payload.accept_id || "" : payload.prompt_id || "");
			const count = Number(payload.chunk_count);
			const index = Number(payload.chunk_index);
			if (!promptId || !Number.isInteger(count) || !Number.isInteger(index) || count < 1 || index < 0 || index >= count) continue;
			let group = roomChunksRef.current.get(promptId);
			if (!group) {
				group = { expected: count, values: Array(count).fill(""), meta: { ...payload, event_type: event.type } };
				roomChunksRef.current.set(promptId, group);
			}
			if (group.expected !== count) continue;
			group.values[index] = String(payload.data_chunk || "");
			if (group.values.some((value) => !value)) continue;
			roomChunksRef.current.delete(promptId);
			roomApplyQueueRef.current = roomApplyQueueRef.current.then(async () => {
				try {
					const delta: Region = {
						bbox: group.meta.bbox,
						shape: group.meta.shape,
						encoding: "zlib-base64-uint8",
						data: group.values.join(""),
					};
					const decoded = await decodeRegion(delta);
					if (group.meta.event_type === "interactive.accept") {
						if (localAcceptedEventsRef.current.has(promptId)) {
							localAcceptedEventsRef.current.delete(promptId);
							return;
						}
						const acceptedVoxels = applyInteractiveCommit(delta.bbox, delta.shape, decoded,
							Number(group.meta.label_id), Boolean(group.meta.allow_overwrite));
						setMessage(`Room member accepted ${group.meta.organ_label || "an organ"} (${acceptedVoxels.toLocaleString()} voxels).`);
					} else {
						if (Number.isInteger(Number(group.meta.label_id))) {
							roomPromptTargetRef.current = { id: Number(group.meta.label_id), name: String(group.meta.organ_label || "") };
						}
						await start();
						setInteractivePreviewRegion(delta.bbox, delta.shape, decoded);
						setMessage(`A room member updated the ${group.meta.organ_label || "selected organ"} AI preview. Accept it to add it to the working labelmap.`);
					}
				} catch (error) {
					setStatus("error");
					setMessage(error instanceof Error ? error.message : "Could not load the shared preview");
				}
			});
		}
	}, [room?.metadata.room_id, room?.pendingEvents, start]);

	return { enabled, status, message, sessionId, start, submit, undo, reset, accept, close, organStatus };
}
