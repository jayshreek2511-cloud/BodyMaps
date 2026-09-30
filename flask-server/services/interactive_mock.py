"""Deterministic, CPU-only stand-in for nnInteractive's remote session.

Enable it only for local development with ``NNINTERACTIVE_MOCK=true``.  It
implements the small part of the remote session API used by BodyMaps, keeping
the HTTP endpoints and viewer flow testable without a model or GPU.
"""

from __future__ import annotations

import numpy as np


class FakeInteractiveSession:
    """A small sphere/box predictor with one-step undo and prompt replay."""

    license = "CC BY-NC-SA 4.0 (mock mode; no model weights loaded)"
    supports_undo = True

    def __init__(self, *_args, **_kwargs):
        self.image = None
        self.target_buffer = None
        self._prompts: list[tuple[str, dict]] = []
        self._undo: tuple[np.ndarray, list[tuple[str, dict]]] | None = None
        self.closed = False

    def ping(self, timeout: float = 5.0) -> bool:
        return not self.closed

    def set_image(self, image: np.ndarray) -> None:
        array = np.asarray(image)
        if array.ndim != 4 or array.shape[0] != 1:
            raise ValueError("image must have shape [1, X, Y, Z]")
        self.image = array

    def set_target_buffer(self, target_buffer: np.ndarray) -> None:
        if self.image is None or tuple(target_buffer.shape) != tuple(self.image.shape[1:]):
            raise ValueError("target buffer must match the image [X, Y, Z] shape")
        self.target_buffer = target_buffer

    def add_initial_seg_interaction(self, initial_seg: np.ndarray, run_prediction: bool = False, **_kwargs) -> None:
        if self.target_buffer is None or tuple(initial_seg.shape) != tuple(self.target_buffer.shape):
            raise ValueError("initial mask must match the image shape")
        self.target_buffer[...] = np.asarray(initial_seg, dtype=np.uint8)

    def _snapshot(self) -> None:
        if self.target_buffer is None:
            raise RuntimeError("set image and target buffer before adding prompts")
        self._undo = (self.target_buffer.copy(), list(self._prompts))

    def _paint_box(self, bounds, include: bool) -> list[list[int]]:
        shape = self.target_buffer.shape
        lo = [max(0, int(pair[0])) for pair in bounds]
        hi = [min(shape[axis], int(bounds[axis][1])) for axis in range(3)]
        if any(hi[d] <= lo[d] for d in range(3)):
            return [[0, 0]] * 3
        slices = tuple(slice(lo[d], hi[d]) for d in range(3))
        if include:
            self.target_buffer[slices] = 1
        else:
            self.target_buffer[slices] = 0
        return [[lo[d], hi[d]] for d in range(3)]

    def _apply(self, kind: str, prompt: dict, include: bool) -> list[list[int]]:
        self._snapshot()
        saved = {**prompt, "include": bool(include)}
        self._prompts.append((kind, saved))
        if kind == "point":
            center = [int(v) for v in prompt["coordinates"]]
            radius = max(2, min(8, min(self.target_buffer.shape) // 24))
            lo = [max(0, v - radius) for v in center]
            hi = [min(self.target_buffer.shape[d], v + radius + 1) for d, v in enumerate(center)]
            axes = [np.arange(lo[d], hi[d]) for d in range(3)]
            mesh = np.ogrid[tuple(slice(v[0], v[-1] + 1) for v in axes)]
            # A deterministic small 3-D sphere, clipped to the image bounds.
            region = sum((mesh[d] - center[d]) ** 2 for d in range(3)) <= radius**2
            view = tuple(slice(lo[d], hi[d]) for d in range(3))
            if include:
                self.target_buffer[view] |= region.astype(np.uint8)
            else:
                self.target_buffer[view][region] = 0
            return [[lo[d], hi[d]] for d in range(3)]
        if kind == "bbox":
            return self._paint_box(prompt["bounds"], include)
        if kind in {"scribble", "lasso"}:
            bounds = prompt["interaction_bbox"]
            shape = self.target_buffer.shape
            lo = [max(0, int(pair[0])) for pair in bounds]
            hi = [min(shape[d], int(bounds[d][1])) for d in range(3)]
            changed = [[lo[d], hi[d]] for d in range(3)]
            target_slice = tuple(slice(lo[d], hi[d]) for d in range(3))
            crop = np.asarray(prompt["crop"], dtype=bool)
            if crop.shape != self.target_buffer[target_slice].shape:
                self.undo()
                raise ValueError("crop shape must match interaction_bbox")
            region = self.target_buffer[target_slice]
            if include:
                region[crop] = 1
            else:
                region[crop] = 0
            return changed
        raise ValueError(f"unsupported prompt type: {kind}")

    def add_point_interaction(self, coordinates, include_interaction=True, **_kwargs):
        return self._apply("point", {"coordinates": list(coordinates)}, include_interaction)

    def add_bbox_interaction(self, bounds, include_interaction=True, **_kwargs):
        return self._apply("bbox", {"bounds": bounds}, include_interaction)

    def add_scribble_interaction(self, crop, include_interaction=True, interaction_bbox=None, **_kwargs):
        return self._apply("scribble", {"crop": np.asarray(crop).copy(), "interaction_bbox": interaction_bbox}, include_interaction)

    def add_lasso_interaction(self, crop, include_interaction=True, interaction_bbox=None, **_kwargs):
        return self._apply("lasso", {"crop": np.asarray(crop).copy(), "interaction_bbox": interaction_bbox}, include_interaction)

    def undo(self) -> bool:
        if self._undo is None or self.target_buffer is None:
            return False
        previous_mask, previous_prompts = self._undo
        self.target_buffer[...] = previous_mask
        self._prompts = previous_prompts
        self._undo = None
        return True

    def reset_interactions(self) -> None:
        if self.target_buffer is not None:
            self.target_buffer.fill(0)
        self._prompts.clear()
        self._undo = None

    def replay(self, prompts: list[dict]) -> None:
        """Replay sanitized wire prompts after the remote lease is re-created."""
        for prompt in prompts:
            kind = prompt["type"]
            include = bool(prompt.get("include", True))
            if kind == "point":
                self.add_point_interaction(prompt["coordinates"], include)
            elif kind == "bbox":
                self.add_bbox_interaction(prompt["bounds"], include)
            elif kind == "scribble":
                self.add_scribble_interaction(prompt["crop"], include, prompt["interaction_bbox"])
            elif kind == "lasso":
                self.add_lasso_interaction(prompt["crop"], include, prompt["interaction_bbox"])
            else:
                raise ValueError(f"unsupported prompt type: {kind}")

    def close(self) -> None:
        self.closed = True
        self.image = None
        self.target_buffer = None
        self._prompts.clear()
        self._undo = None
