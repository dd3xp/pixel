# Every-15-min progress watchdog. Registered as Windows scheduled task
# PixelWatch; survives Claude sessions, VSCode, and reboots.
#
# Scope, by the user's standing order of 2026-09-02: node03 GPU3 ONLY.
# ljq's node09 is never probed and never used, idle or not.
#
# Appends one line per tick to logs_local\watch.log; raises a Windows toast
# when something needs the user: a new baseline result, a FAILING marker, or
# the worker's supervisor gone.  When the baseline queue drains (32/32 and the
# worker exited), it auto-ignites the distillation chain on the same GPU3 --
# the queue takes days, and nobody needs to be awake for the handover.
$ErrorActionPreference = 'SilentlyContinue'
$log = 'C:\Codes\pixel\logs_local\watch.log'
$state = 'C:\Codes\pixel\logs_local\watch.state'

function Toast($title, $msg) {
  try {
    [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
    $xml = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent(
      [Windows.UI.Notifications.ToastTemplateType]::ToastText02)
    $t = $xml.GetElementsByTagName('text')
    $t.Item(0).AppendChild($xml.CreateTextNode($title)) | Out-Null
    $t.Item(1).AppendChild($xml.CreateTextNode($msg)) | Out-Null
    [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('PixelWatch').Show(
      [Windows.UI.Notifications.ToastNotification]::new($xml))
  } catch { }
}

$n03 = ssh -o BatchMode=yes -o ConnectTimeout=30 emnlp "cd /mnt/data/kw/RoundSquisheen/pixel/pixel; echo R=`$(ls baseline/results | wc -l); echo S=`$(pgrep -cf 'supervise[.]sh (bq0|distill2)'); [ -f logs/bq0.FAILING ] || [ -f logs/distill2.FAILING ] && echo FAILING=1 || echo FAILING=0; echo G=`$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3)" 2>$null

$res  = [int](($n03 | Select-String '^R=').ToString() -replace 'R=', '')
$sup  = [int](($n03 | Select-String '^S=').ToString() -replace 'S=', '')
$fail = ($n03 | Select-String '^FAILING=1') -ne $null
$g3   = (($n03 | Select-String '^G=').ToString() -replace 'G=', '')
$line = "[{0}] results={1}/32 sup={2} failing={3} gpu3={4}MiB" -f (Get-Date -Format 'MM-dd HH:mm'), $res, $sup, $fail, $g3
Add-Content -Path $log -Value $line

$prev = @{ res = -1 }
if (Test-Path $state) { $prev = Import-Clixml $state }
if ($n03 -eq $null) { Toast 'PixelWatch' 'node03 ssh 不可达' }
elseif ($fail)      { Toast 'PixelWatch' '出现 FAILING 标记, 需要人看' }
elseif ($res -gt $prev.res -and $prev.res -ge 0) { Toast 'PixelWatch' ("基线新完成: {0}/32" -f $res) }

if ($res -ge 32 -and $sup -lt 1) {
  $ign = ssh -o BatchMode=yes -o ConnectTimeout=30 emnlp "cd /mnt/data/kw/RoundSquisheen/pixel/pixel; [ -f logs/distill2.done ] && { echo HAVE_DONE; exit 0; }; tmux new-session -d -s px2 'setsid nohup bash supervise.sh distill2 3 bash baseline/run_distill_v2.sh </dev/null >/dev/null 2>&1 & disown; sleep 5'; echo IGNITED" 2>$null
  if ("$ign" -match 'IGNITED') { Toast 'PixelWatch' '基线 32/32 收官, 蒸馏链已在 GPU3 自动点火' }
}
elseif ($sup -lt 1 -and $res -lt 32) { Toast 'PixelWatch' 'bq0 监工不在了但队列未跑完, 去看看' }
@{ res = $res } | Export-Clixml $state
