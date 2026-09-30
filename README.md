# Backend

## Live Rooms

Dataset viewer supports temporary, account-free collaborative review at `/live/<room-id>#<room-key>`. Up to eight equal editors can collaborate for 24 hours, then room files expire. Exports include edited labelmap, annotations, notes, chat, event history, and report without changing canonical dataset files.

Deployment and local service instructions: [`flask-server/deploy/LIVE_ROOMS.md`](flask-server/deploy/LIVE_ROOMS.md).

## Interactive segmentation

Interactive 3D segmentation is an opt-in research feature. It uses a separate
GPU service and keeps model previews separate from the working labelmap until
accepted. Setup, local mock flow, proxy requirements, quotas, and licensing are
documented in [`flask-server/deploy/nninteractive.md`](flask-server/deploy/nninteractive.md);
the GPU service compose file is [`docker-compose.nninteractive.yml`](docker-compose.nninteractive.yml).
Select an organ from the viewer's existing label list to seed it from its current
mask. Point mode uses left-click for positive prompts, right-click or Alt-click
for negative prompts, and drag for a box. Accept applies the proposal to the
selected label through viewer undo history and keeps the same volume session
open for switching to another organ. Other organs are protected from overwrite
unless “Allow overwriting other organs” is enabled. Enter accepts, Escape resets
the preview to the organ's seed, Ctrl/Cmd+Z undoes the last prompt, and R resets
the current prompts.

#### Create Conda Environment
```
conda create -n PanTS_backend python=3.11
conda activate PanTS_backend
```

#### Set up environment backend
```
cd flask-server
touch .env  # creates the .env file
nano .env
```

Inside .env file (see `flask-server/.env.example` for the full list):
```
BASE_PATH=/

PANTS_PATH=/folder/where/PanTS

USE_SSL=false
```

Password reset emails (`/api/auth/forgot-password`) and account verification
emails (`/api/auth/send-verification` — verifying is what, with a complete
profile, lifts an account to the 10-scans/day tier) go out over SMTP:

```
# Amazon SES (the production setup — the credential is a send-only SMTP key,
# handed to the deployer privately; it never lives in this repository)
SMTP_HOST=email-smtp.us-east-1.amazonaws.com
SMTP_PORT=587
SMTP_STARTTLS=true
SMTP_USER=<SES SMTP username>
SMTP_PASSWORD=<SES SMTP password>
SMTP_FROM=no-reply@thebodymaps.com
PUBLIC_BASE_URL=https://bodymaps.wse.jhu.edu
```

`PUBLIC_BASE_URL` is what the emailed links point at, so it must be the address
users actually visit. `SMTP_FROM` must be an address on a domain the SES
identity covers. (Gmail SMTP also works for low volume — `smtp.gmail.com`,
port 587, a 16-character App Password rather than the account password — but
its ~500 recipients/day cap and suspension behaviour make it unsuitable once
signups carry verification mail.)

Leave `SMTP_USER`/`SMTP_PASSWORD` unset and both flows still work end to end:
the message — reset or verification link and all — is printed to the server log
instead of being sent, and password-signup users simply stay on the 1-scan/day
tier until mail is configured. That is the intended local-dev path — no mailbox
needed. OAuth (Google/GitHub) accounts arrive already verified and are
unaffected either way.

Optional dataset vars:
```
# Writable dir for precomputed PanTS low-res volumes (make_lowres.py output)
PANTS_LOWRES_PATH=/home/visitor/pants_lowres

# CancerVerse (second, CT-only dataset). Leave unset to disable it.
CANCERVERSE_PATH=/folder/where/CancerVerse
CANCERVERSE_LOWRES_PATH=/home/visitor/cancerverse_lowres
```
`CANCERVERSE_PATH` holds the `CV_########/ct.nii.gz` cases; the metadata CSV `CancerVerse_dataset_metadata.csv` sits **next to** that folder (in its parent). When set, `/api/search?dataset=cancerverse` (or `dataset=all`) searches it; CancerVerse has no masks yet, so mask endpoints return `{"masks_available": false}`.

#### Build the search/shuffle quality index

Search and shuffle can rank cases by CT file size and thumbnail display quality without
adding filesystem or vision-model work to API requests. Generate the index offline on
the host that has the dataset mounted:

```
ollama pull qwen3-vl:4b
python scripts/compute_case_quality.py \
  --out /home/visitor/data/bodymaps_case_quality.v1.json \
  --datasets all \
  --vision required \
  --overwrite
```

Then set the same path in `flask-server/.env` before starting the backend:

```
BODYMAPS_CASE_QUALITY_MANIFEST=/home/visitor/data/bodymaps_case_quality.v1.json
BODYMAPS_THUMBNAIL_VISION_MODEL=qwen3-vl:4b
```

Use `--resume` instead of `--overwrite` to reuse unchanged CT and thumbnail records.
`--vision required` fails rather than silently producing an index without vision
classification. If the manifest is unset or unavailable, ranking falls back to scan
shape and voxel-spacing metadata.

#### Install the visitor-location database

The analytics page maps where visitors come from, using MaxMind's GeoLite2
database on this server — no visitor IP is ever sent to a third party. The file
is ~60MB and not redistributable, so it is not in the repo and each deploy
fetches its own copy. Get a free licence key at
https://www.maxmind.com/en/geolite2/signup (Account > Manage License Keys), then:

```
export MAXMIND_LICENSE_KEY=<your key>
python scripts/download_geolite.py
```

It writes `flask-server/data/GeoLite2-City.mmdb` (override with `GEOIP_DB_PATH`).
Re-run it every month or two — a stale database gets quietly less accurate
rather than failing. Skip this entirely and the site works fine: locations
simply aren't recorded and the map says so.

Behind nginx, also set `TRUST_PROXY=true`. Without it every request looks like
it came from the proxy, so every visitor geolocates to the server itself.

Run backend:

```
pip install -r requirements.txt
python app.py
```

# Frontend

```
cd PanTS-Demo
touch .env
nano .env
```

Inside .env:
```
VITE_API_BASE=http://localhost:5001
```

#### Run frontend

```
npm install
npm run dev
```

# Deploying Updates to the Server
---

After pushing changes, SSH into the server and run the following.

```
ssh visitor@bdmap1.wse.jhu.edu
```

#### One-paste deploy (recommended)

Paste this single line; it pulls the latest `main` and runs the tracked deploy
script, which performs every numbered step below - backup, pull, build,
migrations, restart - with waits and health checks built in:

```
cd /home/visitor/PanTS-Viewer && git fetch && git checkout main && git pull --ff-only && bash flask-server/deploy/deploy.sh
```

If anything goes wrong it stops immediately, **names the step that failed**,
and touches nothing after it - fix the problem and either paste the same line
again (every step is safe to re-run) or continue by hand from that numbered
step below. `bash flask-server/deploy/deploy.sh check` runs the preflight only
and changes nothing.

# The materials below are NOT useful anymore!

#### 1. Back up the database
Accounts, sessions and job state live in SQLite, so take a copy before any deploy that might run migrations.
```
cd /home/visitor/PanTS-Viewer/flask-server && cp *.db ~/db-backup-$(date +%F-%H%M).db 2>/dev/null && echo "backed up" || echo "no db found"
```

#### 2. Pull latest changes
Production must always deploy from `main`. Confirm the branch first, then pull.
```
cd /home/visitor/PanTS-Viewer
git fetch
git checkout main
git pull
```
If `git pull` (or the checkout) refuses because of "local changes would be overwritten," someone edited files directly on the server. Do **not** force past it. Run `git status` to see what changed, then discard each file with `git checkout -- <file>` (or ask the maintainer) before pulling again. The server should never carry local edits.

### Volume-delivery configuration (administrator required)

The viewer's large immutable CT and mask files must be streamed by nginx, not
held open by Gunicorn. Once per server (or whenever the dataset paths change),
an administrator should install the tracked nginx configuration, test it, and
reload nginx:

```
sudo cp /home/visitor/PanTS-Viewer/flask-server/deploy/nginx-bodymaps.conf /etc/nginx/sites-available/bodymaps
sudo ln -sfn /etc/nginx/sites-available/bodymaps /etc/nginx/sites-enabled/bodymaps
sudo nginx -t && sudo systemctl reload nginx
```

Then add `BODYMAPS_ACCEL_REDIRECT_ENABLED=true` to
`/home/visitor/PanTS-Viewer/flask-server/.env`. Do not enable that setting
before `nginx -t` succeeds: the application intentionally falls back to normal
Flask delivery until the private nginx locations exist. A CDN cache rule for
`/api/get-main-nifti/*` and `/api/get-segmentations/*` should also respect the
existing `Cache-Control: public, max-age=604800, immutable` response header;
that is what makes this architecture scale beyond one origin server.

#### 3. Rebuild the frontend and refresh backend dependencies
```
cd /home/visitor/PanTS-Viewer/PanTS-Demo && npm ci && npm run build
/home/visitor/.conda/envs/PanTS_backend/bin/pip install -r /home/visitor/PanTS-Viewer/flask-server/requirements.txt
```
The `pip install` is a fast no-op when nothing changed, but it is required whenever a PR adds or bumps a Python dependency — otherwise the restarted backend crashes on a missing import and the site goes empty. If `npm run build` errors out, **stop here**: nginx keeps serving the old site until a build succeeds, so fix the error before restarting the backend.

#### 4. Apply database migrations
Required whenever a PR adds an Alembic revision; a fast no-op otherwise. Skipping it after a schema change leaves the backend querying tables that do not exist.
```
cd /home/visitor/PanTS-Viewer/flask-server && /home/visitor/.conda/envs/PanTS_backend/bin/alembic upgrade head
```
This step is **mandatory** for the tiered-access release: it carries two
revisions (`7d4f8c2a9b1e` profile fields on `user_account`, and
`9e6a1b5d4c2f` the `email_verification_token` table). Without it, signing up
and editing the profile return 500s. (The earlier account-recovery release's
two revisions run in the same command if the server ever skipped them.)

#### 4b. One-time for the tiered-access release: email credentials (optional)
Verification emails need the SES SMTP block in
`/home/visitor/PanTS-Viewer/flask-server/.env` (the variables are listed in
the email section near the top of this file). **The credential is not in this
repository and never will be** — the maintainer sends a one-time link whose
contents you paste into the SSH terminal as a single block; it appends the
credentials, restarts the backend, and health-checks it. No editor needed.
Skipping this is safe: the site
runs, verification links go to the gunicorn log instead of inboxes, and
password-signup users simply stay on the 1-scan/day tier until it is added.
OAuth sign-ins are unaffected either way.

#### 5. Restart the backend
```
# Stop the old gunicorn process and wait for the port to free
pkill -f "gunicorn.*app:app"; sleep 2
pgrep -f "gunicorn.*app:app" && echo "still running - rerun the line above" || echo "port clear"

# Start a new gunicorn process
nohup /home/visitor/.conda/envs/PanTS_backend/bin/gunicorn \
  --worker-class gthread --workers 1 --threads 8 \
  --bind 127.0.0.1:8000 --timeout 3600 \
  --chdir /home/visitor/PanTS-Viewer/flask-server \
  app:app > /tmp/gunicorn.log 2>&1 &
echo "PID: $!"
```

#### 6. Verify the backend is running
Give it a few seconds to load, then check the backend booted, the dataset loads, and masks serve (all three must succeed).
```
sleep 8
curl http://127.0.0.1:8000/api/ping
curl -s "http://127.0.0.1:8000/api/search?limit=1" | head -c 120; echo
curl -s -o /dev/null -w "segmentations: %{http_code}\n" "http://127.0.0.1:8000/api/get-segmentations/17.nii.gz"
```
Expect `{"message":"pong"}`, a JSON object with `items`, and `segmentations: 200`. If the backend fails to boot, check the log for the traceback.

Then check sign-in is wired up.
```
curl -s http://127.0.0.1:8000/api/auth/oauth/providers
curl -sS -i https://bodymaps.wse.jhu.edu/api/auth/oauth/google | grep -i "^location:" | tr '&' '\n' | grep redirect_uri
```
Expect `{"github":true,"google":true}`, then a `redirect_uri` beginning `https%3A%2F%2Fbodymaps.wse.jhu.edu`. A `http://` scheme there means `PUBLIC_BASE_URL` is unset or stale, and every sign-in fails `redirect_uri_mismatch` — the provider matches that string exactly. Note `providers` only reports whether credentials are non-empty, so it still returns `true` for rotated-but-not-updated secrets; those surface as `invalid_client` at sign-in.

Then check password reset can actually send. This asks for a link for an
address with no account, so nothing is emailed to anyone — it answers 200 either
way, and the point is what the log says next.
```
curl -s -X POST http://127.0.0.1:8000/api/auth/forgot-password \
  -H 'Content-Type: application/json' -d '{"email":"nobody@example.com"}'
grep -i "\[mail\]" /tmp/gunicorn.log | tail -5
```
Expect `{"ok":true}` and **no** `[mail]` lines. A line saying "SMTP is not
configured" means `SMTP_USER`/`SMTP_PASSWORD` are unset and emailed links are
going to the log instead of to users. "SMTP rejected the credentials" means
the password is wrong for the host: for SES it must be the derived SMTP
password (not the raw secret key); for Gmail, an App Password (not the
account password).

Then check the access tiers landed (all three lines matter):
```
# Guests may browse but not upload or use the assistant - expect 401 twice
curl -s -o /dev/null -w "guest upload: %{http_code} (expect 401)\n" -X POST http://127.0.0.1:8000/api/upload-inference-chunk
curl -s -o /dev/null -w "guest assistant: %{http_code} (expect 401)\n" -X POST -H 'Content-Type: application/json' -d '{"message":"hi"}' http://127.0.0.1:8000/api/ai-command

# A fresh account starts on 1 scan/day and gets a verification email
curl -s -c /tmp/tiercheck.txt -X POST -H 'Content-Type: application/json' \
  -d '{"email":"tier-check@example.com","password":"delete-me-123"}' \
  http://127.0.0.1:8000/api/auth/register > /dev/null
curl -s -b /tmp/tiercheck.txt http://127.0.0.1:8000/api/me/usage | head -c 120; echo
```
Expect `"plan": "free"` with `"daily_scans": 1`, and (with SMTP configured) a
verification mail in the inbox — without SMTP, the link is in
`/tmp/gunicorn.log`. Delete the throwaway account afterwards from
Settings → People as an admin, or leave it; it can do nothing but browse.

Finally, load `https://bodymaps.wse.jhu.edu/upload` in a browser: a
signed-out visitor is asked to sign in before uploading (this changed in the
tiered-access release - guests browse, accounts upload), and a signed-in
account can select a file and run.

Logs are written to `/tmp/gunicorn.log`.
