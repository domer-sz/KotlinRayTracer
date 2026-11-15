# Fetches NVIDIA's OptiX SDK headers into native/optix/include.
#
# The headers are NVIDIA's and may not be redistributed, so this repository does not carry them
# (see README.md beside this script). Running the script is accepting NVIDIA's licence for
# them, which it leaves in LICENSE.txt.
param([string]$Tag = "v9.0.0")

$ErrorActionPreference = "Stop"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
# The archive is unpacked in a folder of its own, away from the repository.
$work = Join-Path ([IO.Path]::GetTempPath()) ("optix-dev-" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $work | Out-Null
try {
    $zip = Join-Path $work "optix-dev.zip"
    Invoke-WebRequest -Uri "https://github.com/NVIDIA/optix-dev/archive/refs/tags/$Tag.zip" -OutFile $zip -UseBasicParsing
    Expand-Archive -Path $zip -DestinationPath (Join-Path $work "unpacked")
    $root = Get-ChildItem (Join-Path $work "unpacked") -Directory | Select-Object -First 1
    if (-not (Test-Path (Join-Path $root.FullName "include\optix.h"))) { throw "the archive has no include\optix.h" }

    $target = Join-Path $here "include"
    if (Test-Path $target) { Remove-Item -Recurse -Force $target }
    Copy-Item -Recurse (Join-Path $root.FullName "include") $target
    Copy-Item (Join-Path $root.FullName "LICENSE.txt") (Join-Path $here "LICENSE.txt") -Force
    Set-Content -Path (Join-Path $here "VERSION") -Value $Tag -Encoding ascii
    "OptiX headers $Tag are in $target"
} finally {
    Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
}
