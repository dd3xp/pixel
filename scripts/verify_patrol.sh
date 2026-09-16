#!/bin/bash
# Wait for any running patrol to finish, then fire the SCHEDULED TASK once and report whether its
# context can authenticate (the direct run works; the task context previously returned 403).
cd /c/Codes/pixel
for i in $(seq 1 240); do [ -f logs_local/.patrol_running ] || break; sleep 15; done
before=$(ls -t logs_local/patrol_*.log 2>/dev/null | head -1)
powershell.exe -NoProfile -Command "Start-ScheduledTask -TaskName PixelPatrol" >/dev/null 2>&1
sleep 40
for i in $(seq 1 100); do [ -f logs_local/.patrol_running ] || break; sleep 15; done
after=$(ls -t logs_local/patrol_*.log 2>/dev/null | head -1)
if [ "$after" = "$before" ]; then echo "TASK_PRODUCED_NO_LOG"; exit 1; fi
if grep -qi "403\|Failed to authenticate\|forbidden" "$after"; then
  echo "TASK_AUTH_FAILED: $(head -c 200 "$after")"; exit 1
fi
echo "TASK_OK: $after ($(stat -c %s "$after") bytes)"; tail -6 "$after"
