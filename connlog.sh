#!/bin/bash
# Log which machines are connected to this shared account, continuously.
#
# The sentinel identifies a killer only if the sender is still alive when the
# signal is handled -- and a `bash -c 'kill -TERM ...'` can be gone microseconds
# later (one capture already came back "sender already exited").  A SIGKILL
# would leave nothing at all, since it cannot be caught.
#
# This does not depend on catching the sender: everyone on this box shares uid
# 1001, so `ss -tnp` attributes every ssh socket on the account, including other
# people's.  Sampling the peer IPs every few seconds means that when a kill
# lands at time T, the log already says which machines were attached around T.
#
# Read-only: it observes sockets and never touches anyone's processes.
set -u
ROOT=/mnt/data/kw/RoundSquisheen/pixel/pixel
LOG=$ROOT/logs/connections.log
INTERVAL=${INTERVAL:-5}
prev=""

while true; do
  # incoming ssh sessions have the local port 22; strip our own noise later by
  # comparing against the IP this project connects from
  now=$(ss -tnp state established '( sport = :22 )' 2>/dev/null \
        | awk 'NR>1 {print $4"  <-  "$5}' | sort -u)
  if [ "$now" != "$prev" ]; then
    {
      echo "[$(date +%m-%d\ %H:%M:%S)] ssh sessions changed:"
      if [ -z "$now" ]; then echo "    (none)"; else echo "$now" | sed 's/^/    /'; fi
    } >> "$LOG"
    prev="$now"
  fi
  sleep "$INTERVAL"
done
