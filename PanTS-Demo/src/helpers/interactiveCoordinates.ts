import { roundedIJK, type IJK } from "./interactiveSegmentation";

/**
 * Cornerstone supplies LPS world points; worldToIndex maps them through the
 * loaded CT affine into NIfTI/model array order [i,j,k]. No plane-based axis
 * permutation or display-resolution scaling is applied.
 */
export function viewerWorldToNiftiIJK(
	worldLps: readonly number[],
	worldToIndex: (world: readonly number[]) => readonly number[],
	dimensions: readonly number[],
): IJK {
	return roundedIJK(worldToIndex(worldLps), dimensions);
}
