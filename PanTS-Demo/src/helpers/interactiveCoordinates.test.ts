import { describe, expect, it } from "vitest";
import { viewerWorldToNiftiIJK } from "./interactiveCoordinates";
import { mergeInteractiveLabel } from "./interactiveSegmentation";

describe("viewer-to-NIfTI interactive coordinates", () => {
	it("uses the loaded affine result in [i,j,k] order without plane permutations", () => {
		const calls: number[][] = [];
		const ijk = viewerWorldToNiftiIJK(
			[12, -4, 20],
			(world) => {
				calls.push([...world]);
				return [181.4, 44.6, 190];
			},
			[503, 324, 223],
		);
		expect(calls).toEqual([[12, -4, 20]]);
		expect(ijk).toEqual([181, 45, 190]);
	});

	it("rejects affine results outside the original CT grid", () => {
		expect(() => viewerWorldToNiftiIJK([0, 0, 0], () => [503, 0, 0], [503, 324, 223])).toThrow("outside");
	});
});

describe("interactive organ replacement", () => {
	it("preserves other labels and clears selected-organ voxels removed by a smaller proposal", () => {
		const old = [0, 15, 15, 27];
		const nextProposal = [1, 1, 0, 0];
		const accepted = old.map((current, index) => mergeInteractiveLabel(current, nextProposal[index], 15));
		expect(accepted).toEqual([15, 15, 0, 27]);
	});
});
