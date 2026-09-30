import { describe, expect, it } from "vitest";
import { interactiveShortcut, maskRegionIndex, mergeInteractiveLabel, roundedIJK, sliceBounds, sliceForPlane } from "./interactiveSegmentation";

describe("interactive segmentation coordinate convention", () => {
	it("keeps Cornerstone and NIfTI [i,j,k] order", () => {
		expect(roundedIJK([4.49, 7.5, 2], [10, 11, 12])).toEqual([4, 8, 2]);
		expect(sliceForPlane("axial")).toBe(2);
		expect(sliceForPlane("sagittal")).toBe(0);
		expect(sliceForPlane("coronal")).toBe(1);
	});

	it("builds a half-open box restricted to the current slice", () => {
		expect(sliceBounds(2, 6, [5, 7, 3], [1, 2, 3])).toEqual([[1, 6], [2, 8], [6, 7]]);
	});

	it("indexes compressed NumPy C-order [i,j,k] regions", () => {
		expect(maskRegionIndex(1, 2, 3, [2, 4, 5])).toBe(33);
	});

	it("rejects a coordinate outside the loaded volume", () => {
		expect(() => roundedIJK([10, 0, 0], [10, 10, 10])).toThrow("outside");
	});

	it("preserves other organs unless overwrite is enabled and clears removed voxels", () => {
		expect(mergeInteractiveLabel(7, 1, 3)).toBe(7);
		expect(mergeInteractiveLabel(7, 1, 3, true)).toBe(3);
		expect(mergeInteractiveLabel(3, 0, 3)).toBe(0);
		expect(mergeInteractiveLabel(7, 0, 3)).toBe(7);
	});

	it("maps the documented keyboard shortcuts", () => {
		expect(interactiveShortcut("Enter")).toBe("accept");
		expect(interactiveShortcut("Escape")).toBe("discard");
		expect(interactiveShortcut("z", true)).toBe("undo");
		expect(interactiveShortcut("z", false, true)).toBe("undo");
		expect(interactiveShortcut("r")).toBe("reset");
		expect(interactiveShortcut("x")).toBeNull();
	});
});
