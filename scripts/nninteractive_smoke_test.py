#!/usr/bin/env python3
"""Run a one-case live nnInteractive coordinate/Dice smoke test.

Requires NNINTERACTIVE_SERVER_URL, NNINTERACTIVE_API_KEY (if configured by the
server), PANTS_PATH, and a PanTS case with its original CT and combined labels.
The script is intentionally opt-in and never writes dataset files.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


def dice(prediction: np.ndarray, truth: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    denominator = int(prediction.sum()) + int(truth.sum())
    return 1.0 if denominator == 0 else 2.0 * float(np.logical_and(prediction, truth).sum()) / denominator


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_id", nargs="?", help="PanTS numeric ID or PanTS_########")
    parser.add_argument("--dataset-root", default=os.environ.get("PANTS_PATH"), help="PanTS dataset root (defaults to PANTS_PATH)")
    parser.add_argument("--label-id", type=int, help="Ground-truth label value; defaults to the largest non-background organ")
    args = parser.parse_args()

    server_url = os.environ.get("NNINTERACTIVE_SERVER_URL", "").strip()
    if not server_url:
        print("SKIP: NNINTERACTIVE_SERVER_URL is not configured")
        return 0
    if not args.case_id:
        parser.error("case_id is required when a server is configured")
    if not args.dataset_root:
        parser.error("Set PANTS_PATH or pass --dataset-root")

    case = args.case_id
    folder = case if case.startswith("PanTS_") else f"PanTS_{int(case):08d}"
    root = Path(args.dataset_root).resolve()
    ct_path = root / "image_only" / folder / "ct.nii.gz"
    label_path = root / "mask_only" / folder / "combined_labels.nii.gz"
    if not ct_path.is_file() or not label_path.is_file():
        raise FileNotFoundError(f"Expected original CT and labels at {ct_path} and {label_path}")

    image = np.ascontiguousarray(np.asanyarray(nib.load(str(ct_path), mmap=True).dataobj))
    labels = np.asanyarray(nib.load(str(label_path), mmap=True).dataobj)
    if image.ndim != 3 or labels.shape != image.shape:
        raise ValueError(f"CT and label dimensions must match; got {image.shape} and {labels.shape}")
    label_id = args.label_id
    if label_id is None:
        ids, counts = np.unique(labels[labels > 0], return_counts=True)
        if not len(ids):
            raise ValueError("No non-background organ labels are present in the ground-truth mask")
        label_id = int(ids[int(np.argmax(counts))])
    organ = labels == label_id
    voxels = np.argwhere(organ)
    if not len(voxels):
        raise ValueError(f"Ground-truth label {label_id} is empty")
    center = voxels.mean(axis=0)
    point = voxels[np.argmin(np.square(voxels - center).sum(axis=1))].astype(int)

    try:
        from nnInteractive.inference.remote import nnInteractiveRemoteInferenceSession
    except ImportError as exc:
        raise RuntimeError("Install flask-server requirements to use nninteractive-client") from exc

    api_key = os.environ.get("NNINTERACTIVE_API_KEY")

    def run(coordinates: list[int]) -> float:
        kwargs = {"server_url": server_url}
        if api_key:
            kwargs["api_key"] = api_key
        session = nnInteractiveRemoteInferenceSession(**kwargs)
        try:
            session.set_image(image[np.newaxis, ...])
            target = np.zeros(image.shape, dtype=np.uint8)
            session.set_target_buffer(target)
            session.add_point_interaction(coordinates, include_interaction=True)
            result = getattr(session, "target_buffer", target)
            if callable(result):
                result = result()
            return dice(np.asarray(result), organ)
        finally:
            session.close()

    ijk = [int(v) for v in point]
    swapped = [ijk[1], ijk[0], ijk[2]]
    correct_score = run(ijk)
    swapped_score = run(swapped)
    higher = "IJK [i,j,k]" if correct_score > swapped_score else "swapped i/j" if swapped_score > correct_score else "tie"
    print(f"case={folder} label_id={label_id} point_ijk={ijk}")
    print(f"Dice IJK [i,j,k]: {correct_score:.6f}")
    print(f"Dice swapped [j,i,k]: {swapped_score:.6f}")
    print(f"Higher score: {higher}")
    if swapped_score > correct_score:
        print("WARNING: swapped coordinates scored higher; verify the viewer/model axis convention.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
