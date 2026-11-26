<#
启动独立 MocapStudio 处理工作台（仅监听 127.0.0.1；不依赖采集网站或 Node 服务器）。
用法：
  .\启动MocapStudio.ps1
  .\启动MocapStudio.ps1 -Build
  .\启动MocapStudio.ps1 -Port 8787 -MaxWorkers 2
根路径 / 自动跳转 /workbench/；工作台仅在用户提交任务后启动私有 worker。
Ctrl+C 关闭服务并回收本服务拥有的 worker，不终止其他 Shogun/Blender 会话。
#>
[CmdletBinding()]
param(
    [int]$Port = 8787,
    [string]$StateDir,
    [string[]]$DataRoot,
    [string]$RepoScripts,
    [string]$ShogunCl,
    [string]$Blender,
    [ValidateRange(1, 4)][int]$MaxWorkers = 1,
    [string[]]$AllowOrigin,
    [switch]$Build
)

$ErrorActionPreference = 'Stop'
if ($RepoScripts) { $env:MOCAP_REPO_SCRIPTS = $RepoScripts }
$workbench = Join-Path $PSScriptRoot 'workbench'
if ($Build) {
    Push-Location $workbench
    try {
        & npm.cmd ci
        if ($LASTEXITCODE -ne 0) { throw "npm ci failed ($LASTEXITCODE)" }
        & npm.cmd run build
        if ($LASTEXITCODE -ne 0) { throw "Workbench build failed ($LASTEXITCODE)" }
    }
    finally { Pop-Location }
}
if (-not (Test-Path -LiteralPath (Join-Path $workbench 'dist\index.html') -PathType Leaf)) {
    throw '工作台尚未构建。请运行 .\启动MocapStudio.ps1 -Build；构建后启动只需要 Python，不需要 Node。'
}
$serverArgs = @(
    (Join-Path $PSScriptRoot 'server\app_server.py'),
    '--port', "$Port",
    '--max-workers', "$MaxWorkers"
)
if ($StateDir) { $serverArgs += @('--state-dir', $StateDir) }
if ($ShogunCl) { $serverArgs += @('--shogun-cl', $ShogunCl) }
if ($Blender) { $serverArgs += @('--blender', $Blender) }
foreach ($root in $DataRoot) { $serverArgs += @('--data-root', $root) }
foreach ($origin in $AllowOrigin) { $serverArgs += @('--allow-origin', $origin) }
Write-Host "MocapStudio 工作台：http://127.0.0.1:$Port/"
Write-Host '仅限本机；请勿通过反向代理公开。处理写入独立状态目录，原始输入保留。'
& python @serverArgs
exit $LASTEXITCODE
