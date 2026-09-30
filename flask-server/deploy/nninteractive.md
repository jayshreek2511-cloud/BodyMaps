# nnInteractive service

BodyMaps runs nnInteractive as a separate GPU service. Flask is the only
client; the browser never receives the shared server key. The server owns the
model and GPU. Flask owns verified-account checks, per-user quotas, volume data,
room authorization, prompt replay, and session cleanup.

## GPU host

Install Docker Engine and the NVIDIA Container Toolkit, set a strong key in the
GPU host's environment, then start the checked-in compose file:

```bash
export NN_INTERACTIVE_API_KEY="$(openssl rand -hex 32)"
export NNINTERACTIVE_MAX_SESSIONS=4
docker compose -f docker-compose.nninteractive.yml up -d
curl http://127.0.0.1:1527/healthz
```

The compose file binds to loopback. If Flask runs on another machine, bind the
published port to the GPU host's private interface and allow TCP 1527 only from
the Flask host. Keep the API off the public internet. The upstream service
supports separate client sessions over one loaded model; the GPU still runs
predictions serially. Raise `--max-sessions` to accommodate active users, not
to increase GPU throughput. A second GPU server is the upstream scaling path.

## Flask host

Add these settings to `flask-server/.env` and restart Gunicorn:

```dotenv
NNINTERACTIVE_ENABLED=true
NNINTERACTIVE_SERVER_URL=http://127.0.0.1:1527
NNINTERACTIVE_API_KEY=<same private key as the GPU host>
NNINTERACTIVE_MAX_SESSIONS_PER_USER=2
NNINTERACTIVE_DAILY_QUOTA=5
NNINTERACTIVE_IDLE_SECONDS=900
NNINTERACTIVE_MOCK=false
```

Set the server URL to the GPU host's private address when the processes run on
different machines. Do not put the API key in Vite environment variables or
browser settings. The API returns 404 while `NNINTERACTIVE_ENABLED` is false.
Only logged-in, verified researcher accounts can start or use sessions. Rooms
reuse one serialized session per room; each verified editor is charged against
their own daily session quota the first time they use it.

The Flask manager is process-local. Production currently runs one Gunicorn
worker with threads, which is supported. If worker count is increased, route a
given user/room consistently to the same worker or move session state, prompt
logs, buffers, and per-session locks into a shared session service. Merely
increasing `--workers` will make a session appear missing when the next request
lands on another worker.

## Reverse proxy

If a proxy sits between Flask and the GPU server, keep it private and pass the
nnInteractive client headers through unchanged. The client uses the bearer
`Authorization` key and a per-session `X-Lease-Token`; `X-Meta` carries session
metadata. Prediction responses stream compressed mask deltas, so turn response
buffering off and allow enough time for large volumes:

```nginx
location / {
    proxy_pass http://nninteractive_upstream;
    proxy_http_version 1.1;
    proxy_set_header Authorization $http_authorization;
    proxy_set_header X-Lease-Token $http_x_lease_token;
    proxy_set_header X-Meta $http_x_meta;
    proxy_set_header Content-Type $http_content_type;
    proxy_request_buffering off;
    proxy_buffering off;
    proxy_read_timeout 180s;
    proxy_send_timeout 180s;
}
```

Do not expose this proxy to browser clients. Flask sends the shared bearer key
server-side. The upstream client handles lease tokens and heartbeats.

## Local mock

For a CPU-only end-to-end development flow, keep the feature enabled and set
`NNINTERACTIVE_MOCK=true`. `services/interactive_mock.py` replaces the remote
session with deterministic point, box, scribble, and lasso masks. It does not
load nnInteractive or model weights. Use a verified local account and a local
PanTS case. Click **HD** first, open **Annotate**, choose an Interactive tool,
place prompts, then accept the preview into the working labelmap. In a room,
each accepted prompt preview is broadcast and retained in the room event
export as provenance.

## Organ refinement workflow

Select an existing organ in the viewer's segment list, then choose **Segment
organ** or an Interactive point/box/scribble/lasso tool. The viewer label ID and
color stay attached to that organ; no label name or numeric ID is entered. An
existing canonical organ mask seeds the session, and a label without a mask
starts empty. Positive and negative point prompts, current-slice box prompts,
scribbles, and lasso outlines update a separate preview. **Accept** applies it
through the viewer's undo history and leaves the original image session open.
Switching the selected organ replaces the prompt state and loads that organ's
seed without reloading the CT. By default, a proposal leaves other organ IDs
untouched; enable **Allow overwriting other organs** only when that is intended.
Accepted shrinkage clears removed voxels belonging to the selected organ.

Point mode: left click adds a positive point; right click or Alt-click adds a
negative point; dragging draws a box. Enter accepts, Escape restores the seed,
Ctrl/Cmd+Z undoes the last prompt, and R resets prompts. Live Room exports store
the organ label/ID, prompt count, prompt details, and preview delta as provenance.

## Real-model coordinate smoke test

On a host with access to the GPU service and local original PanTS files, set
`PANTS_PATH`, `NNINTERACTIVE_SERVER_URL`, and (if required) the server's
`NNINTERACTIVE_API_KEY`, then run:

```bash
python scripts/nninteractive_smoke_test.py 8854
```

The script reads the original CT and `combined_labels.nii.gz`, chooses a
positive voxel from the largest organ unless `--label-id` is supplied, predicts
once using `[i,j,k]` and once with `i/j` swapped, and prints both Dice scores.
It does not alter either NIfTI file. It exits successfully with a `SKIP` message
when `NNINTERACTIVE_SERVER_URL` is unset.

## Limits and licensing

The feature is opt-in and intended for research use. The upstream model weights
are CC BY-NC-SA 4.0; the maintainers should confirm the planned deployment and
use comply with that license before enabling it for users. Review the upstream
[server/client deployment guide](https://github.com/MIC-DKFZ/nnInteractive/blob/master/SERVER_CLIENT.md)
and [Docker guide](https://github.com/MIC-DKFZ/nnInteractive/blob/master/nnInteractive/inference/server/DOCKER.md)
when upgrading the image or model version.
