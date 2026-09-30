import type { MaskRange, SharedMeasurement, SharedMprView } from "../helpers/CornerstoneNifti2";

export type LiveRoomMetadata = {
	room_id: string;
	case_id: string;
	resolution: "low" | "full";
	created_at: string;
	expires_at: string;
	geometry_hash: string;
	dimensions: number[];
	latest_seq: number;
	mode: "review" | "quiz";
	quiz_pack_id?: string;
	quiz_timer_seconds?: number | null;
};

export type LiveQuizPhase = "lobby" | "question_open" | "question_closed" | "question_revealed" | "completed";

export type LiveQuizChoice = { id: string; label: string };
export type LiveQuizQuestion = {
	id: string;
	prompt: string;
	choices: LiveQuizChoice[];
	viewer_cue?: {
		clear_overlays?: boolean;
		crosshair_lps?: [number, number, number];
	};
};
export type LiveQuizConsistency = {
	status: "consistent" | "inconsistent" | "incomplete";
	reasons: string[];
};
export type LiveQuizLeaderboardRow = {
	rank: number;
	participant_id: string;
	name: string;
	score: number;
	max_score: number;
	total_response_ms: number;
	consistency: LiveQuizConsistency;
};
export type LiveQuizReveal = {
	question_id: string;
	correct_choice_id: string;
	explanation: string;
	source_label?: string;
	distribution: Record<string, number>;
	viewer_cue?: {
		show_lesion_overlay?: boolean;
		crosshair_lps?: [number, number, number];
		reference_measurement_lps?: [[number, number, number], [number, number, number]];
		reference_diameter_mm?: number;
		lesion_label?: number;
	};
};
export type LiveQuizState = {
	revision?: number;
	phase: LiveQuizPhase;
	question_index: number;
	question_count: number;
	current_question: LiveQuizQuestion | null;
	deadline_at: string | null;
	remaining_seconds: number | null;
	timer_paused: boolean;
	response_count: number;
	eligible_count: number;
	reveal: LiveQuizReveal | null;
	leaderboard: LiveQuizLeaderboardRow[];
	consistency_summary: Record<"consistent" | "inconsistent" | "incomplete", number>;
	round_completed: boolean;
	host_connected: boolean;
};
export type LiveQuizHostCredential =
	| { mode: "modern"; value: string }
	| { mode: "legacy"; value: string };
export type LiveQuizSubmission = {
	choice_id: string;
	answered_at: string;
	response_ms: number;
	name: string;
};

export type LiveRoomMeasurement = SharedMeasurement & { revision?: number; _seq?: number };

export type LiveRoomNote = {
	id: string;
	author: string;
	text: string;
	world: [number, number, number];
	plane: string;
	organ_label?: string;
	revision?: number;
	_seq?: number;
};

export type LiveRoomChatMessage = {
	id: string;
	author: string;
	text: string;
	timestamp?: string;
	_seq?: number;
};

export type LiveRoomDurableState = {
	measurements: Record<string, LiveRoomMeasurement>;
	notes: Record<string, LiveRoomNote>;
	chat: LiveRoomChatMessage[];
	undone_event_ids?: string[];
};

export type LiveRoomMaskPatch = {
	operation_id: string;
	geometry_hash: string;
	resolution: "low" | "full";
	segment_label: number;
	ranges: MaskRange[];
};

export type LiveRoomEvent = {
	seq: number;
	event_id: string;
	type: "measurement.upsert" | "measurement.delete" | "mask.patch" | "note.upsert" | "note.delete" | "chat.add" | "interactive.prompt" | "interactive.accept"
		| "quiz.started" | "quiz.closed" | "quiz.revealed" | "quiz.advanced"
		| "quiz.host_paused" | "quiz.host_resumed" | "quiz.host_promoted";
	participant_id: string;
	name: string;
	created_at: string;
	payload: Record<string, unknown>;
	undo_of?: string;
};

export type LiveRoomEventDelivery = {
	event: LiveRoomEvent;
	replayed: boolean;
};

export type LiveRoomParticipant = {
	participant_id: string;
	name: string;
	color: string;
	role: "reviewer" | "host" | "student";
	cursor?: { pane: string; x: number; y: number } | null;
	crosshair?: [number, number, number] | null;
	active_tool?: string | null;
	target_organ?: string | null;
	plane?: string | null;
	following?: string | null;
	view?: (SharedMprView & {
		windowWidth?: number;
		windowCenter?: number;
		opacity?: number;
		visibleOrgans?: boolean[];
	}) | null;
};

export type LiveRoomDurableCallbacks = {
	onRejected?: () => void;
};

export type LiveRoomConnectionState = "connecting" | "connected" | "reconnecting" | "disconnected" | "expired" | "error";

export type LiveRoomController = {
	metadata: LiveRoomMetadata;
	roomKey: string;
	maskUrl: string;
	participantId: string;
	name: string;
	connectionState: LiveRoomConnectionState;
	participants: LiveRoomParticipant[];
	state: LiveRoomDurableState;
	pendingEvents: LiveRoomEventDelivery[];
	acknowledgeEvents: (throughSequence: number) => void;
	followingId: string | null;
	error: string | null;
	undoNotice: string | null;
	quiz: LiveQuizState | null;
	quizOwnSubmissions: Record<string, LiveQuizSubmission>;
	quizEligible: boolean;
	isHost: boolean;
	collaborationLocked: boolean;
	sendDurable: (
		type: LiveRoomEvent["type"],
		payload: Record<string, unknown>,
		eventId?: string,
		callbacks?: LiveRoomDurableCallbacks
	) => Promise<boolean>;
	sendPresence: (payload: Record<string, unknown>) => void;
	sendView: (payload: Record<string, unknown>) => void;
	sendChat: (text: string) => Promise<boolean>;
	addNote: (text: string, world: [number, number, number], plane: string, organLabel?: string) => Promise<boolean>;
	deleteNote: (id: string) => Promise<boolean>;
	requestUndo: () => void;
	follow: (participantId: string) => void;
	stopFollowing: () => void;
	copyShareLink: () => Promise<void>;
	downloadExport: (kind: "zip" | "pdf") => Promise<void>;
	startQuiz: () => boolean;
	answerQuiz: (choiceId: string) => boolean;
	closeQuiz: () => boolean;
	revealQuiz: () => boolean;
	advanceQuiz: () => boolean;
};
