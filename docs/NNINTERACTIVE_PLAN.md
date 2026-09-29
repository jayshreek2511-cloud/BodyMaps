# nnInteractive integration plan

## Scope and phases

1. **Plan and local mock:** document the data flow and coordinate contract; provide a deterministic fake remote session that can be selected without installing or running a GPU model.
2. **Backend sessions and endpoints:** add a feature-flagged blueprint, per-user/per-room remote sessions, full-resolution image loading, prompt validation, compressed region deltas, commit without writing canonical dataset files, auth/verification/quota checks, cleanup, and mocked service tests.
3. **Viewer tools:** add the API client, coordinate conversion helper and tests, preview overlay, prompt controls, accept/undo/reset, status and license disclosure; keep all controls hidden when the feature is disabled.
4. **Live Rooms:** serialize room prompts, commit/broadcast prediction deltas through the existing room event stream, and include prompt provenance in room exports.
5. **Operations:** document environment settings, GPU Docker deployment, nginx proxy requirements, worker-count constraints, and an optional deploy preflight; add a README link.

## Files expected to change

- Backend: `flask-server/app.py`, `flask-server/constants.py`, `flask-server/requirements.txt`, `flask-server/.env.example`, `flask-server/api/interactive.py`, `flask-server/services/interactive_sessions.py`, `flask-server/services/nninteractive_predictor.py`, `flask-server/services/plan_store.py`, `flask-server/models/usage_event.py`, `flask-server/migrations/env.py`, and a new Alembic revision only if existing usage bookkeeping cannot safely represent the separate quota.
- Backend tests: new unit and functional tests under `flask-server/tests/` for mock and remote adapter behavior, validation, authorization, quota, server errors, recovery, deltas, and canonical-file immutability.
- Frontend: a new API/helper module and coordinate helper under `PanTS-Demo/src/helpers/`, interactive controls and preview UI under `PanTS-Demo/src/components/` and `PanTS-Demo/src/routes/VisualizationPage.tsx`, plus focused Vitest coverage.
- Live Rooms and deploy: `flask-server/live_rooms_ws.py`, `flask-server/services/live_room_store.py`, `flask-server/deploy/deploy.sh`, new `flask-server/deploy/nninteractive.md`, root `docker-compose.nninteractive.yml`, and `README.md`.
- Local testing: fake session implementation in `flask-server/services/` selectable with `NNINTERACTIVE_MOCK=true`; this avoids any GPU, model download, or server setup.

## Coordinate and array convention

The nnInteractive remote API expects a four-dimensional image shaped `[C, X, Y, Z]`; its target mask is three-dimensional `[X, Y, Z]`. The BodyMaps NIfTI loader uses nibabel's voxel array order `(i, j, k)` and the current Cornerstone volume reports IJK dimensions in the same voxel-axis order. Thus the model coordinate `[x, y, z]` is the viewer voxel IJK `[i, j, k]` after converting a pointer's world LPS millimetres through the NIfTI affine (`LPS -> RAS -> inverse affine -> nearest voxel`). It is not an array `z,y,x` transpose. Screen axes vary by MPR plane, so the frontend helper must use the active pane's world-to-index transform and pin the pane's slice axis before serializing a 2D prompt. Every emitted bbox/crop must have exactly one axis of length one, use half-open bounds, and be checked against the original volume shape.

For Live Room masks, the existing durable mask-patch protocol uses flat C-order offsets over the viewer's `[X,Y,Z]` label array. The server's `[X,Y,Z]` prediction delta must therefore be flattened with the same C order before it is encoded into room mask ranges.

## Assumptions and constraints

- `NNINTERACTIVE_ENABLED` remains false by default. A second `NNINTERACTIVE_MOCK` switch is only for local development and tests; it is ignored unless the feature is explicitly enabled.
- Flask owns every real remote session and is the only component that knows the remote URL and API key. The browser calls Flask with the existing same-origin/API-base conventions.
- Only logged-in, email-verified accounts with a complete verified-researcher profile (the existing Pro eligibility rule) can start sessions. Quota is a separate rolling 24-hour allowance, independent of scan inference quota.
- A manager session is scoped to one user and one dataset/case, or to one capability-key Live Room. Every session has a serialization lock. The deployment currently uses one Gunicorn worker; an in-memory manager cannot be shared across workers, so multi-worker deployment requires a shared session coordinator/sticky routing or moving the manager into a dedicated service.
- CT input is the original full-resolution NIfTI volume read without windowing, normalization, or uint8 conversion. Only returned delta masks use uint8.
- Preview masks remain separate from the canonical dataset labelmap until accepted. User acceptance writes only to an owned session/room working mask; canonical dataset files are immutable.
- Model weights are CC BY-NC-SA 4.0; the UI and deployment notes disclose research-use restrictions. Maintainers must confirm their deployment and use comply with that license before enabling the feature publicly.
- Expired remote sessions are recoverable only when the server supports session creation and prompt replay; replay is bounded to the stored prompt log and all operations remain serialized.
