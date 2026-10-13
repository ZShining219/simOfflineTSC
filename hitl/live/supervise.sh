#!/usr/bin/env bash
# Long-running supervisor for the SUMO live broadcast web service.
#
# The web/management layer stays resident; the SUMO world is only loaded
# while a session is open from the page.  This wrapper restarts the service
# process if it exits unexpectedly.
#
# Usage:
#   supervise.sh start [--port N] [-- ARGS...]   # launch under supervision
#   supervise.sh stop                            # stop supervisor + service
#   supervise.sh status                          # show current state
#   supervise.sh restart                         # restart the service process
#
# Environment override: LIVE_PORT (default 8010), LIVE_PY (python binary).

set -u
REPO=/projects/simOfflineTSC
PY=${LIVE_PY:-/home/dev/miniforge3/envs/colight/bin/python}
PORT=${LIVE_PORT:-8010}
LOG=/tmp/sumo_live_broadcast.log
SUPLOG=/tmp/sumo_live_supervise.log
PIDFILE=/tmp/sumo_live_broadcast.pid
SUPPIDFILE=/tmp/sumo_live_supervise.pid

# pgrep -f self-matches the caller's own command line, so the supervisor is
# tracked by pidfile and verified via /proc cmdline.
_svc_pid() {
  for p in $(pgrep -f "hitl.live.*port.*${PORT}"); do
    [ "$(cat /proc/$p/comm 2>/dev/null)" = "python" ] && echo "$p" && return 0
  done
  return 1
}
_sup_pid() {
  [ -f "$SUPPIDFILE" ] || return 1
  local p; p=$(cat "$SUPPIDFILE" 2>/dev/null)
  [ -n "$p" ] && [ -d "/proc/$p" ] \
    && grep -qa "supervise.sh" "/proc/$p/cmdline" 2>/dev/null \
    && echo "$p"
}

case "${1:-start}" in
  start)
    shift
    if [ -n "$(_sup_pid)" ]; then
      echo "supervisor already running (pid $(_sup_pid))"
      exit 0
    fi
    if [ -n "$(_svc_pid)" ]; then
      echo "service already running without supervisor (pid $(_svc_pid))"
      exit 0
    fi
    setsid nohup bash "$0" _loop "$@" >>"$SUPLOG" 2>&1 < /dev/null &
    echo $! > "$SUPPIDFILE"
    echo "supervisor started (pid $(cat "$SUPPIDFILE")), log: $SUPLOG / $LOG"
    ;;
  _loop)
    # Internal: the actual supervision loop (do not call directly).
    shift
    echo $$ > "$SUPPIDFILE"
    echo "$(date '+%F %T') supervisor loop start (port=$PORT)" >>"$SUPLOG"
    while true; do
      cd "$REPO" || exit 1
      "$PY" -u -m hitl.live --port "$PORT" "$@" >>"$LOG" 2>&1
      rc=$?
      echo "$(date '+%F %T') service exited (rc=$rc); restarting in 3s" >>"$SUPLOG"
      sleep 3
    done
    ;;
  stop)
    sup=$(_sup_pid)
    [ -n "$sup" ] && kill "$sup" && echo "supervisor stopped ($sup)"
    svc=$(_svc_pid)
    [ -n "$svc" ] && kill "$svc" && echo "service stopped ($svc)"
    rm -f "$PIDFILE" "$SUPPIDFILE"
    ;;
  restart)
    svc=$(_svc_pid)
    [ -n "$svc" ] && kill "$svc" && echo "service killed ($svc); supervisor will restart it"
    [ -z "$svc" ] && echo "service not running" && "$0" start
    ;;
  status)
    sup=$(_sup_pid); svc=$(_svc_pid)
    echo "supervisor: ${sup:-not running}"
    echo "service:    ${svc:-not running}"
    [ -n "$svc" ] && tail -n 3 "$LOG" 2>/dev/null
    ;;
  *)
    echo "usage: $0 {start|stop|restart|status} [-- ARGS]" >&2
    exit 2
    ;;
esac
