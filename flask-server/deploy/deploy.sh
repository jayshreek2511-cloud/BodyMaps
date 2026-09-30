#!/bin/bash
# One-paste deploy for bdmap1. Runs every step of the README's "Deploying
# Updates to the Server" section with waits and checks built in, and stops at
# the first failure NAMING the step - so the fallback is simply to continue
# manually from that numbered step in the README.
#
# Safe to re-run from the top at any time: every step is a no-op when its work
# is already done. `bash deploy.sh check` runs the preflight only and changes
# nothing.
#
# The paths are bdmap1's; the BODYMAPS_* variables exist so the script can be
# rehearsed on another machine and are never needed on the server.
set -u

REPO="${BODYMAPS_REPO:-/home/visitor/PanTS-Viewer}"
PY="${BODYMAPS_CONDA_BIN:-/home/visitor/.conda/envs/PanTS_backend/bin}"
PORT="${BODYMAPS_PORT:-8000}"
LOG="${BODYMAPS_GUNICORN_LOG:-/tmp/gunicorn.log}"

step=""
say() { printf '\n== %s ==\n' "$*"; }
die() {
  printf '\nDEPLOY FAILED at: %s\n' "$step"
  [ -n "${1:-}" ] && printf '%s\n' "$1"
  printf 'Nothing after this step ran. Fix it, then rerun this script (safe),\nor continue manually from the README step named above.\n'
  exit 1
}

step="preflight"
say "$step"
[ -d "$REPO/.git" ] || die "repo not found at $REPO - is this the right machine?"
for bin in gunicorn pip alembic python; do
  [ -x "$PY/$bin" ] || die "missing $PY/$bin - conda env not where expected"
done
# npm often lives behind nvm, which interactive shells load from ~/.bashrc but
# a script does not inherit. Find it before giving up: source nvm directly,
# then glob the usual install dirs, then ask a login+interactive shell.
if ! command -v npm >/dev/null 2>&1; then
  if [ -s "$HOME/.nvm/nvm.sh" ]; then
    . "$HOME/.nvm/nvm.sh" >/dev/null 2>&1 || true
  fi
fi
if ! command -v npm >/dev/null 2>&1; then
  npm_dir=$(ls -d "$HOME"/.nvm/versions/node/*/bin /usr/local/node*/bin /opt/node*/bin 2>/dev/null | sort -V | tail -1)
  if [ -n "$npm_dir" ] && [ -x "$npm_dir/npm" ]; then
    export PATH="$npm_dir:$PATH"
  fi
fi
if ! command -v npm >/dev/null 2>&1; then
  found=$(bash -ilc 'command -v npm' 2>/dev/null | tail -1)
  if [ -n "$found" ] && [ -x "$found" ]; then
    export PATH="$(dirname "$found"):$PATH"
  fi
fi
command -v npm >/dev/null 2>&1 || die "npm not found anywhere. In your SSH session run:  bash -ilc 'command -v npm'  and send the output to the maintainer."
echo "npm: $(command -v npm)"
command -v curl >/dev/null 2>&1 || die "curl is not on PATH"
free_kb=$(df -Pk "$REPO" | awk 'NR==2 {print $4}')
[ "${free_kb:-0}" -ge 2097152 ] || die "less than 2 GB free disk at $REPO"
echo "preflight ok (repo, conda env, npm, curl, disk)"
if [ -f "$REPO/flask-server/.env" ] && grep -Eq '^NNINTERACTIVE_ENABLED=true([[:space:]]|$)' "$REPO/flask-server/.env"; then
  nninteractive_url="${NNINTERACTIVE_SERVER_URL:-}"
  if [ -z "$nninteractive_url" ]; then
    nninteractive_url=$(sed -n 's/^NNINTERACTIVE_SERVER_URL=["\x27]*\([^"\x27[:space:]]*\).*/\1/p' "$REPO/flask-server/.env" | tail -1)
  fi
  nninteractive_url="${nninteractive_url:-http://127.0.0.1:1527}"
  if curl -fsS --max-time 5 "${nninteractive_url%/}/healthz" >/dev/null; then
    echo "nnInteractive reachable at $nninteractive_url"
  else
    echo "WARNING: nnInteractive is enabled but $nninteractive_url/healthz did not respond (continuing preflight)"
  fi
fi
if [ "${1:-}" = "check" ]; then
  echo "check mode: stopping here - nothing was changed."
  exit 0
fi

step="1. back up the database (README step 1)"
say "$step"
cd "$REPO/flask-server" || die
ts=$(date +%F-%H%M%S)
found=0
for db in *.db; do
  [ -e "$db" ] || continue
  cp "$db" "$HOME/db-backup-$ts-$db" || die "backup copy failed"
  found=1
done
if [ "$found" = 1 ]; then echo "backed up to ~/db-backup-$ts-*"; else echo "no db found (first deploy?) - continuing"; fi

step="2. pull latest main (README step 2)"
say "$step"
cd "$REPO" || die
git fetch || die "git fetch failed - network or credentials"
dirty=$(git status --porcelain --untracked-files=no)
if [ -n "$dirty" ]; then
  printf '%s\n' "$dirty"
  die "the server carries local edits (listed above). Per the README, do NOT force past this: discard each with 'git checkout -- <file>' or ask the maintainer, then rerun."
fi
git checkout main >/dev/null 2>&1 || die "could not switch to main"
old_sha=$(git rev-parse --short HEAD)
git pull --ff-only || die "pull was not a fast-forward - history diverged; ask the maintainer"
new_sha=$(git rev-parse --short HEAD)
echo "main: $old_sha -> $new_sha"

step="3. rebuild frontend and backend deps (README step 3)"
say "$step"
cd "$REPO/PanTS-Demo" || die
npm ci || die "npm ci failed"
npm run build || die "frontend build failed. nginx keeps serving the old site and the backend was NOT restarted - fix the build error and rerun."
"$PY/pip" install -r "$REPO/flask-server/requirements.txt" || die "pip install failed"
echo "frontend built; backend deps refreshed"

step="4. database migrations (README step 4)"
say "$step"
cd "$REPO/flask-server" || die
"$PY/alembic" upgrade head || die "alembic failed. The pre-migration backup from step 1 is at ~/db-backup-$ts-*"
echo "migrations at head"

step="5. restart the backend (README step 5)"
say "$step"
pkill -f "gunicorn.*app:app" 2>/dev/null
for i in 1 2 3 4 5 6 7 8 9 10; do
  pgrep -f "gunicorn.*app:app" >/dev/null 2>&1 || break
  sleep 1
done
if pgrep -f "gunicorn.*app:app" >/dev/null 2>&1; then
  pkill -9 -f "gunicorn.*app:app"; sleep 2
fi
pgrep -f "gunicorn.*app:app" >/dev/null 2>&1 && die "the old gunicorn refused to exit"
nohup "$PY/gunicorn" \
  --worker-class gthread --workers 1 --threads 8 \
  --bind "127.0.0.1:$PORT" --timeout 3600 \
  --chdir "$REPO/flask-server" \
  app:app > "$LOG" 2>&1 &
echo "started gunicorn (PID $!), log: $LOG"

step="6. verify (README step 6)"
say "$step"
up=0
for i in $(seq 1 30); do
  curl -sf "http://127.0.0.1:$PORT/api/ping" >/dev/null 2>&1 && { up=1; break; }
  sleep 2
done
[ "$up" = 1 ] || die "backend did not answer /api/ping within 60s - traceback is in $LOG"
echo "ping ok"
curl -s "http://127.0.0.1:$PORT/api/search?limit=1" | grep -q '"items"' || die "search endpoint broken - see $LOG"
echo "search ok"
seg_code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/api/get-segmentations/17.nii.gz")
if [ "$seg_code" = "200" ]; then echo "segmentations ok (200)"; else echo "WARNING: segmentations returned $seg_code (dataset paths not mounted? not fatal for the deploy)"; fi
gate_code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "http://127.0.0.1:$PORT/api/upload-inference-chunk")
[ "$gate_code" = "401" ] || die "guest upload returned $gate_code, expected 401 - the access tiers did not deploy"
echo "guest gate ok (401)"
jar=$(mktemp)
# SES mailbox simulator: accepts the verification mail without a bounce, so
# deploy checks never hurt sender reputation. Harmless when SMTP is unset
# (the message just goes to the log like any other).
throwaway="success+deploy-$ts@simulator.amazonses.com"
curl -s -c "$jar" -X POST -H 'Content-Type: application/json' \
  -d "{\"email\":\"$throwaway\",\"password\":\"delete-me-$ts\"}" \
  "http://127.0.0.1:$PORT/api/auth/register" >/dev/null || die "register failed - see $LOG"
curl -s -b "$jar" "http://127.0.0.1:$PORT/api/me/usage" | "$PY/python" -c '
import json, sys
d = json.load(sys.stdin)
assert d.get("plan") == "free", "fresh account plan is %r, expected free" % d.get("plan")
assert d["limits"]["daily_scans"] == 1, "daily_scans is %r, expected 1" % d["limits"]["daily_scans"]
print("tier ok: fresh account on free, 1 scan/day")
' || die "tier check failed - see the assertion above and $LOG"
rm -f "$jar"
echo "(throwaway account $throwaway can be deleted from Settings -> People, or ignored)"
if grep -qi "SMTP is not configured" "$LOG" 2>/dev/null; then
  echo "NOTE: SMTP is not configured - verification links go to $LOG instead of inboxes (README step 4b turns real email on; the site works fine meanwhile)"
else
  echo "mail: no 'SMTP is not configured' lines in the log"
fi

printf '\nDEPLOY OK  main %s -> %s\n' "$old_sha" "$new_sha"
