# nnInteractive organ-edit workflow

## Existing work and scope

- The current branch already implements Flask-managed remote sessions, raw full-resolution CT input, per-organ seeding, prompt preview/accept/undo/reset, protected label merges, keyboard shortcuts, an explicit local mock, and a GPU smoke script. This task reuses that work.
- Existing implementation arrived in `c2a2b19` (mock/plan), `2c0e2fe` (backend), `1b160c6` (organ-first viewer), and `f8d3b28` (deployment/smoke); later fixes `56ab8e9` and `220c0e1` were also present before this task.
- Keep `NNINTERACTIVE_ENABLED=false` by default. The server owns the API key; the browser uses the existing Flask endpoints. Canonical dataset files remain read-only.
- This change is limited to route gating, one Organ AI toolbar entry, and directly tested coordinate conversion. Existing brush and eraser remain unchanged.

## Case 4 inspection (2026-10-01)

- Original CT and `combined_labels.nii.gz` are both `(503, 324, 223)`, `int16`, with matching affine and `('L','A','I')` orientation.
- Viewer slice counts are axial `223`, sagittal `503`, coronal `324`; the HD toggle uses the full grid.
- Lung Left is scalar label ID `15`, with `601,912` voxels and bounds `i=181..400, j=44..218, k=0..190`. Its blockiness is present in the stored labelmap, not caused by a low-resolution display resample.
- The selected existing-class row provides the label ID, name, and color. The accepted proposal uses that ID/color and protects other label values; preview is a separate layer.
- Prompt coordinates are Cornerstone LPS world points transformed by the loaded CT's `worldToIndex` to IJK. The NIfTI/model array uses the same `[i,j,k]` order, with no screen-plane permutation.
- The point tool supports positive left-click, negative right-click/Alt-click, and drag-box. It will be surfaced as one “Organ AI” toolbar entry.

## Planned files

- Existing files (5): this plan, `flask-server/app.py`, `flask-server/api/interactive.py`, `PanTS-Demo/src/components/viewer/AnnotationToolbar.tsx`, and `PanTS-Demo/src/helpers/CornerstoneNifti2.tsx`.
- New files: a feature-gated blueprint-registration helper, a pure coordinate helper, and focused backend/frontend tests.

## Verification boundary

- With the feature disabled, the blueprint is not registered and case loading stays on its existing path.
- Verify route gating, coordinate mapping, no-overwrite merges, and shrink-on-reaccept with tests; retain existing backend/frontend checks and TypeScript check.
- Out of scope: local 3D mesh manifests may fail when `MESH_PATH` points to the unrelated Linux default. Do not modify that behavior here.
- Before deployment, a human must configure the private GPU host and API key, confirm the model-weight license, and run `scripts/nninteractive_smoke_test.py` against the real service.
