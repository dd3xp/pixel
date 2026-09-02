# Every-15-min progress watchdog. Registered as Windows scheduled task
# PixelWatch; survives Claude sessions, VSCode, and reboots.
# Appends one line per tick to logs_local\watch.log; raises a Windows toast
# when something needs the user: new baseline results, a FAILING marker, the
# worker's supervisor gone, or node09's 8 cards free (time to scale up).
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

$n03 = ssh -o BatchMode=yes -o ConnectTimeout=30 emnlp "cd /mnt/data/kw/RoundSquisheen/pixel/pixel; echo R=`$(ls baseline/results | wc -l); echo S=`$(pgrep -cf 'supervise[.]sh bq0'); [ -f logs/bq0.FAILING ] && echo FAILING=1 || echo FAILING=0; echo G=`$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3)" 2>$null
$n09 = ssh -o BatchMode=yes -o ConnectTimeout=30 kw "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1; find /mnt/data/kw/ljqinfer_dsv4f_tp8 /mnt/data/kw/wt_24ms -maxdepth 2 -newermt '-60 minutes' 2>/dev/null | wc -l" 2>$null

$res  = [int](($n03 | Select-String '^R=').ToString() -replace 'R=', '')
$sup  = [int](($n03 | Select-String '^S=').ToString() -replace 'S=', '')
$fail = ($n03 | Select-String '^FAILING=1') -ne $null
$g3   = (($n03 | Select-String '^G=').ToString() -replace 'G=', '')
$k9   = [int](@($n09)[0].ToString().Trim())
$act9 = [int](@($n09)[-1].ToString().Trim())   # files he touched in the last hour
$line = "[{0}] results={1}/32 sup={2} failing={3} gpu3={4}MiB node09max={5}MiB act9={6}" -f (Get-Date -Format 'MM-dd HH:mm'), $res, $sup, $fail, $g3, $k9, $act9
Add-Content -Path $log -Value $line

$prev = @{ res = -1; free9 = $false }
if (Test-Path $state) { $prev = Import-Clixml $state }
if ($n03 -eq $null) { Toast 'PixelWatch' 'node03 ssh 不可达' }
elseif ($fail)      { Toast 'PixelWatch' 'bq0 打出 FAILING 标记, 需要人看' }
elseif ($sup -lt 1) { Toast 'PixelWatch' 'bq0 监工不在了 (被杀或跑完), 去看看' }
elseif ($res -gt $prev.res -and $prev.res -ge 0) { Toast 'PixelWatch' ("基线新完成: {0}/32" -f $res) }
# 0 MiB alone is just a gap between his bursts; free means idle memory AND
# no file he touched for an hour.
$free9 = ($n09 -ne $null) -and ($k9 -lt 1000) -and ($act9 -eq 0)
if ($free9 -and -not $prev.free9) { Toast 'PixelWatch' 'node09 八卡已空! 可以扩 worker + 点火蒸馏链' }
@{ res = $res; free9 = $free9 } | Export-Clixml $state
