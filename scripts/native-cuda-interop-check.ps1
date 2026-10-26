[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Executable,
    [int]$Frames = 1000,
    [string]$OutputDirectory = '.native-build/interop/cuda/matrix',
    [int]$TimeoutSeconds = 600,
    [switch]$GpuValidation,
    [switch]$Unchecked,
    [switch]$EdgeOnly
)
$ErrorActionPreference = 'Stop'
if (!$IsWindows -or $Frames -lt 4 -or $Frames -gt 100000 -or $TimeoutSeconds -lt 1) { throw 'Invalid platform/frame count/timeout.' }
$root = Split-Path $PSScriptRoot -Parent
Push-Location $root
try {
    $binary = (Resolve-Path $Executable).Path
    $output = New-Item -ItemType Directory -Force $OutputDirectory
    $dll = Join-Path (Split-Path $binary -Parent) 'oidn_webgpu.dll'
    $cases = @(
        @{ Name = 'small'; Weights = 'weights/rt_hdr_alb_nrm_small.tza'; Size = 16; Tile = 0 },
        @{ Name = 'base'; Weights = 'weights/rt_hdr_alb_nrm.tza'; Size = 16; Tile = 0 },
        @{ Name = 'large'; Weights = 'weights/rt_hdr_calb_cnrm_large.tza'; Size = 31; Tile = 16 }
    )
    if ($EdgeOnly) { foreach ($case in $cases) { $case.Size = 31; $case.Tile = 16 } }
    $reports = [Collections.Generic.List[object]]::new()
    foreach ($case in $cases) {
        foreach ($fp32 in @(0, 1)) {
            foreach ($copy in @(0, 1)) {
                $name = "$($case.Name)-fp$(if ($fp32) {32} else {16})-$(if ($copy) {'copy'} else {'shared'})"
                $arguments = @($case.Weights, "$Frames", "$($case.Size)", "$($case.Tile)", "$fp32", "$copy",
                    $(if ($Unchecked) {'0'} else {'1'}), $(if ($GpuValidation) {'1'} else {'0'}))
                $start = [Diagnostics.ProcessStartInfo]::new($binary)
                $start.UseShellExecute = $false
                $start.CreateNoWindow = $true
                $start.RedirectStandardOutput = $true
                $start.RedirectStandardError = $true
                $start.WorkingDirectory = $root
                foreach ($argument in $arguments) { $start.ArgumentList.Add($argument) }
                $process = [Diagnostics.Process]::Start($start)
                $stdout = $process.StandardOutput.ReadToEndAsync()
                $stderr = $process.StandardError.ReadToEndAsync()
                $timedOut = !$process.WaitForExit($TimeoutSeconds * 1000)
                if ($timedOut) { $process.Kill($true); $process.WaitForExit() }
                $outText = $stdout.GetAwaiter().GetResult()
                $errText = $stderr.GetAwaiter().GetResult()
                [IO.File]::WriteAllText((Join-Path $output.FullName "$name.stdout.json"), $outText)
                [IO.File]::WriteAllText((Join-Path $output.FullName "$name.stderr.log"), $errText)
                $metadata = [ordered]@{
                    arguments = $arguments; executable = $binary; executable_sha256 = (Get-FileHash $binary).Hash
                    dll = $dll; dll_sha256 = (Get-FileHash $dll).Hash
                    weights_sha256 = (Get-FileHash $case.Weights).Hash
                    baseline = (& git rev-parse HEAD); dirty = [bool](& git status --porcelain --untracked-files=normal)
                    sources = @(Get-ChildItem native/cuda/*, native/include/oidn_webgpu_cuda.hpp, native/examples/cuda_dx12_gpu.cu | Get-FileHash | Select-Object Path, Hash)
                    drivers = @(Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion)
                    toolkit = (& nvcc --version) -join "`n"
                    agility_path = $env:WGPU_DX12_AGILITY_SDK_PATH; agility_version = $env:WGPU_DX12_AGILITY_SDK_VERSION
                    utc = [DateTime]::UtcNow.ToString('o'); timed_out = $timedOut; exit_code = $process.ExitCode; report = $null
                }
                if (!$timedOut -and $process.ExitCode -eq 0) {
                    $metadata.report = (@($outText -split '\r?\n' | Where-Object { $_.StartsWith('{') })[-1]) | ConvertFrom-Json
                    if (!$metadata.report.output_byte_equal -or $metadata.report.cpu_image_transfers -ne 0) { throw "$name did not meet the GPU contract." }
                    $wantedPrecision = if ($fp32) {'fp32'} else {'fp16'}
                    if ($metadata.report.precision -ne $wantedPrecision) { throw "$name used the wrong precision." }
                }
                $metadata | ConvertTo-Json -Depth 15 | Set-Content (Join-Path $output.FullName "$name.metadata.json") -Encoding utf8
                $reports.Add($metadata)
                $reports | ConvertTo-Json -Depth 15 | Set-Content (Join-Path $output.FullName 'summary.json') -Encoding utf8
                $process.Dispose()
                if ($timedOut -or $metadata.exit_code -ne 0) { throw "$name failed; inspect $($output.FullName)." }
                Write-Output "$name : $Frames frames, byte equal"
            }
        }
    }
} finally { Pop-Location }
