# Restored from the proven v6_cron.ps1 (git a1a641a) that drove the autonomous
# work 08-16..08-27, then got deleted in 36cc092 as "unused".  A Windows
# scheduled task fires this every 15 min; it launches a FRESH headless Claude
# that reads the log for memory, does one patrol step (check GPU3 jobs, score,
# fix, advance, log, commit), and exits.  OS-level: survives session crashes,
# closed windows, and reboots.  A lock file skips a tick if the previous run
# is still going, so runs never overlap.
$ErrorActionPreference = "Continue"
Set-Location "C:\Codes\pixel"
# Task Scheduler starts clean, but clear the nested-session guard anyway so a
# manual test from inside a Claude session also works.
$env:CLAUDECODE = $null
$env:CLAUDE_CODE_SSE_PORT = $null
$env:CLAUDE_CODE_ENTRYPOINT = $null

$lock = "C:\Codes\pixel\logs_local\.patrol_running"
if (Test-Path $lock) {
  if (((Get-Date) - (Get-Item $lock).LastWriteTime) -lt [TimeSpan]::FromMinutes(30)) {
    Add-Content "C:\Codes\pixel\logs_local\patrol.log" ("[{0}] skip: previous run still active" -f (Get-Date -Format 'MM-dd HH:mm'))
    exit 0
  }
}
New-Item -ItemType File -Force $lock | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$log = "C:\Codes\pixel\logs_local\patrol_$stamp.log"
$prompt = Get-Content -Raw -Encoding UTF8 "C:\Codes\pixel\scripts\patrol_prompt.txt"
Add-Content "C:\Codes\pixel\logs_local\patrol.log" ("[{0}] patrol start -> {1}" -f (Get-Date -Format 'MM-dd HH:mm'), (Split-Path $log -Leaf))
try {
  & claude -p $prompt --dangerously-skip-permissions --output-format text --max-turns 60 2>&1 | Out-File -Encoding utf8 $log
} finally {
  Remove-Item -Force $lock -ErrorAction SilentlyContinue
  Add-Content "C:\Codes\pixel\logs_local\patrol.log" ("[{0}] patrol done" -f (Get-Date -Format 'MM-dd HH:mm'))
}
