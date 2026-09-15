#!/usr/bin/env bash
# Hand the panel to launchd so it keeps running after the terminal closes.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
LABEL="io.conduktor.divoom"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOGDIR="$HOME/Library/Logs"
PYTHON="$HERE/.venv/bin/python"

usage() {
  cat <<'EOF'
usage: ./install-agent.sh install [args...]   # default: logo spin
       ./install-agent.sh stop
       ./install-agent.sh start
       ./install-agent.sh uninstall
       ./install-agent.sh log

Anything after `install` is passed to `python -m divoom.serve`, so:
  ./install-agent.sh install logocycle --each 90
  ./install-agent.sh install screensaver --each 60
  ./install-agent.sh install face doom
EOF
}

[ -x "$PYTHON" ] || { echo "missing $PYTHON — run: uv venv && uv pip install -e ."; exit 1; }
[ -x "$HERE/bin/divoom-bridge" ] || { echo "missing bin/divoom-bridge — run ./build.sh"; exit 1; }

case "${1:-}" in
install)
  shift
  ARGS=("${@:-}")
  [ ${#ARGS[@]} -eq 0 ] || [ -z "${ARGS[0]}" ] && ARGS=(logo spin)

  mkdir -p "$(dirname "$PLIST")" "$LOGDIR"
  {
    echo '<?xml version="1.0" encoding="UTF-8"?>'
    echo '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">'
    echo '<plist version="1.0"><dict>'
    echo "  <key>Label</key><string>$LABEL</string>"
    echo '  <key>ProgramArguments</key><array>'
    printf '    <string>%s</string>\n' "$PYTHON" "-m" "divoom.serve" "${ARGS[@]}"
    echo '  </array>'
    echo "  <key>WorkingDirectory</key><string>$HERE</string>"
    echo '  <key>RunAtLoad</key><true/>'
    # KeepAlive restarts it whatever the exit reason; serve.py has its own
    # backoff, so this only matters if the interpreter itself dies.
    echo '  <key>KeepAlive</key><true/>'
    echo '  <key>ThrottleInterval</key><integer>15</integer>'
    echo '  <key>ProcessType</key><string>Background</string>'
    echo "  <key>StandardOutPath</key><string>$LOGDIR/$LABEL.log</string>"
    echo "  <key>StandardErrorPath</key><string>$LOGDIR/$LABEL.log</string>"
    echo '</dict></plist>'
  } > "$PLIST"

  launchctl bootout "gui/$UID/$LABEL" 2>/dev/null || true
  launchctl bootstrap "gui/$UID" "$PLIST"
  echo "installed: ${ARGS[*]}"
  echo "log: $LOGDIR/$LABEL.log"
  ;;
stop)     launchctl bootout "gui/$UID/$LABEL" && echo stopped ;;
start)    launchctl bootstrap "gui/$UID" "$PLIST" && echo started ;;
uninstall)
  launchctl bootout "gui/$UID/$LABEL" 2>/dev/null || true
  rm -f "$PLIST"
  echo "removed $PLIST"
  ;;
log)      tail -f "$LOGDIR/$LABEL.log" ;;
*)        usage; exit 2 ;;
esac
