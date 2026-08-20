$ErrorActionPreference = "Continue"
Set-Location "C:\Codes\pixel"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$log = "C:\Codes\pixel\logs_local\v6_cron_$stamp.log"
$lock = "C:\Codes\pixel\logs_local\.running"
if (Test-Path $lock) {
  if ((Get-Date) - (Get-Item $lock).LastWriteTime -lt [TimeSpan]::FromMinutes(50)) { "skip: previous run still active" | Out-File $log; exit 0 }
}
New-Item -ItemType File -Force $lock | Out-Null
$prompt = Get-Content -Raw -Encoding UTF8 "C:\Codes\pixel\scripts\v6_cron_prompt.md"
try {
  & claude -p $prompt --dangerously-skip-permissions --output-format text --max-turns 60 2>&1 | Out-File -Encoding utf8 $log
} finally {
  Remove-Item -Force $lock -ErrorAction SilentlyContinue
}
