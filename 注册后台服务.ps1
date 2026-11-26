# Mocap Studio 后台服务注册/启动
# 服务进程经 WMI 创建,父进程为系统 WmiPrvSE —— 不属于任何终端/IDE/dsh 进程树,
# 宿主退出时不会再被作业对象连带杀掉,批处理不会中断。
param(
    [switch]$Unregister,
    [int]$Port = 8787,
    [string]$JobsDir = 'H:\MocapData\jobs',
    [string]$DataRoot = 'H:\MocapData'
)
$ErrorActionPreference = 'Stop'
$pythonw = (Get-Command pythonw.exe).Source
$server  = Join-Path $PSScriptRoot 'server\app_server.py'
if (-not (Test-Path $server)) { throw '找不到 server\app_server.py(请在 MocapStudio 根目录运行)' }
$cmd = '"' + $pythonw + '" "' + $server + '" --port ' + $Port + ' --allow-origin http://127.0.0.1:5173 --jobs-dir "' + $JobsDir + '" --data-root "' + $DataRoot + '"'
$startup = Join-Path ([Environment]::GetFolderPath('Startup')) 'MocapStudioServer.vbs'
if ($Unregister) {
    if (Test-Path $startup) { Remove-Item $startup -Force }
    Write-Host '已移除开机自启(运行中的实例未动)'
    exit 0
}
$vbs = 'Set fso = CreateObject("WScript.Shell")' + [Environment]::NewLine + 'fso.Run "' + ($cmd -replace '"', '""') + '", 0, False'
Set-Content -Encoding Default $startup $vbs
$existing = Get-NetTCPConnection -LocalPort $Port -State Listen -EA SilentlyContinue
if ($existing) { Stop-Process -Id $existing[0].OwningProcess -Force; Start-Sleep 1 }
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd; CurrentDirectory = (Split-Path $server) }
if ($r.ReturnValue -ne 0) { throw 'WMI 创建进程失败 ReturnValue=' + $r.ReturnValue }
for ($i = 0; $i -lt 15; $i++) {
    try {
        $code = (Invoke-WebRequest -UseBasicParsing ("http://127.0.0.1:" + $Port + "/api/processing/config") -TimeoutSec 5).StatusCode
        Write-Host '服务已启动(PID ' + $r.ProcessId + '),HTTP ' + $code + ';开机自启已写入 ' + $startup
        exit 0
    } catch { Start-Sleep 1 }
}
throw '服务未在 15 秒内监听' 
