/** Viewer/NIfTI/model coordinates share IJK = [i, j, k] array-axis order.
 * Cornerstone `worldToIndex` returns IJK and nibabel's proxy array is indexed
 * [i, j, k], so no axis permutation or display-plane reversal is needed. */

export type IJK = [number, number, number];
export type Bounds3 = [[number, number], [number, number], [number, number]];
export type ViewerPlane = "axial" | "sagittal" | "coronal";

export const PLANE_TO_ARRAY_AXIS: Record<ViewerPlane, 0 | 1 | 2> = {
	axial: 2,
	sagittal: 0,
	coronal: 1,
};

export function roundedIJK(values: readonly number[], dimensions: readonly number[]): IJK {
	if (values.length !== 3 || dimensions.length !== 3 || dimensions.some((d) => !Number.isInteger(d) || d <= 0)) {
		throw new Error("A 3D voxel grid is required");
	}
	const point = values.map((value, axis) => {
		if (!Number.isFinite(value)) throw new Error("Voxel coordinates must be finite");
		const index = Math.round(value);
		if (index < 0 || index >= dimensions[axis]) throw new Error("Point is outside the image");
		return index;
	});
	return point as IJK;
}

export function sliceBounds(axis: 0 | 1 | 2, sliceIndex: number, first: IJK, second: IJK): Bounds3 {
	const mins = first.map((value, i) => Math.min(value, second[i]));
	const maxs = first.map((value, i) => Math.max(value, second[i]) + 1);
	mins[axis] = sliceIndex;
	maxs[axis] = sliceIndex + 1;
	return mins.map((min, i) => [min, maxs[i]]) as Bounds3;
}

export function sliceForPlane(plane: ViewerPlane): 0 | 1 | 2 {
	return PLANE_TO_ARRAY_AXIS[plane];
}

export function decodeRegion(data: Uint8Array, shape: readonly number[]): Uint8Array {
	const size = shape.reduce((product, dimension) => product * dimension, 1);
	if (shape.length !== 3 || shape.some((dimension) => !Number.isInteger(dimension) || dimension < 1) || data.length !== size) {
		throw new Error("Interactive mask region has invalid dimensions");
	}
	return data;
}

export function maskRegionIndex(i: number, j: number, k: number, shape: readonly number[]): number {
	return (i * shape[1] + j) * shape[2] + k;
}

/** Merge a binary proposal into one organ without claiming other organs' voxels. */
export function mergeInteractiveLabel(current: number, proposed: number, selectedLabel: number, allowOverwrite = false): number {
	if (proposed) return current === 0 || current === selectedLabel || allowOverwrite ? selectedLabel : current;
	return current === selectedLabel ? 0 : current;
}

export type InteractiveShortcut = "accept" | "discard" | "undo" | "reset" | null;
export function interactiveShortcut(key: string, ctrl = false, meta = false): InteractiveShortcut {
	if ((ctrl || meta) && key.toLowerCase() === "z") return "undo";
	if (key === "Enter") return "accept";
	if (key === "Escape") return "discard";
	if (key.toLowerCase() === "r") return "reset";
	return null;
}
