# Run from the repository root in an x64 VS developer PowerShell (PowerShell 7+).
# GPU correctness only. Logs include deliberate synchronization delays.
[CmdletBinding()]
param(
    [Parameter(Mandatory)][ValidateSet('vulkan', 'dx12')][string]$Backend,
    [int]$Frames = 1000,
    [string]$Adapter = 'RTX 4060',
    [string]$OutputDirectory = '.native-build/interop',
    [int]$TimeoutSeconds = 600,
    [switch]$GpuValidation,
    [switch]$NoDebug,
    [switch]$Unchecked,
    [switch]$SkipBuild
)
$ErrorActionPreference = 'Stop'
if (!$IsWindows) { throw 'The current interop prototypes require Windows.' }
if ($Frames -lt 1 -or $Frames -gt 100000 -or $TimeoutSeconds -lt 1) { throw 'Invalid frame count or timeout.' }
if ($NoDebug -and ($Backend -ne 'dx12' -or $GpuValidation)) { throw '-NoDebug is a DX12 diagnostic mode without GPU validation.' }
$root = Split-Path $PSScriptRoot -Parent
Push-Location $root
try {
    $output = New-Item -ItemType Directory -Force $OutputDirectory
    $example = "${Backend}_interop"
    if (!$SkipBuild) {
        & cargo +stable-x86_64-pc-windows-msvc build --release --locked --target x86_64-pc-windows-msvc --manifest-path native/Cargo.toml --example $example
        if ($LASTEXITCODE -ne 0) { throw 'Cargo build failed.' }
    }
    $binary = (Resolve-Path "native/target/x86_64-pc-windows-msvc/release/examples/$example.exe").Path
    $binaryHash = (Get-FileHash -LiteralPath $binary -Algorithm SHA256).Hash
    if ($GpuValidation) {
        if ($Backend -eq 'vulkan') {
            $env:VK_LAYER_GPUAV_ENABLE = '1'
            $env:VK_LAYER_VALIDATE_SYNC = '1'
        } else { $env:WGPU_GPU_BASED_VALIDATION = '1' }
    }
    $cases = @(
        @{ Name = 'small'; Weights = 'weights/rt_hdr_alb_nrm_small.tza'; Size = 16; Tile = 0 },
        @{ Name = 'base'; Weights = 'weights/rt_hdr_alb_nrm.tza'; Size = 16; Tile = 0 },
        @{ Name = 'large'; Weights = 'weights/rt_hdr_calb_cnrm_large.tza'; Size = 31; Tile = 16 }
    )
    $reports = [Collections.Generic.List[object]]::new()
    foreach ($case in $cases) {
        foreach ($precision in @('fp16', 'fp32')) {
            $arguments = @('--frames', "$Frames", '--size', "$($case.Size)", '--tile', "$($case.Tile)", '--weights', $case.Weights, '--adapter', $Adapter, '--trace')
            if ($precision -eq 'fp32') { $arguments += '--fp32' }
            if ($Backend -eq 'vulkan' -and $case.Name -eq 'base') { $arguments += '--shared-queue' }
            if ($NoDebug) { $arguments += '--no-debug' }
            if ($Unchecked) { $arguments += '--unchecked' }
            $name = "$Backend-$($case.Name)-$precision"
            if ($GpuValidation) { $name += '-gpu-validation' }
            if ($NoDebug) { $name += '-no-debug' }
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
            $errorText = $stderr.GetAwaiter().GetResult()
            [IO.File]::WriteAllText((Join-Path $output.FullName "$name.stdout.json"), $outText)
            [IO.File]::WriteAllText((Join-Path $output.FullName "$name.stderr.log"), $errorText)
            $metadata = [ordered]@{
                prototype = $true; backend = $Backend; case = $case.Name; precision = $precision
                executable = $binary; executable_sha256 = $binaryHash; arguments = $arguments
                weights_sha256 = (Get-FileHash -LiteralPath $case.Weights -Algorithm SHA256).Hash
                exit_code = $process.ExitCode; timed_out = $timedOut
                utc = [DateTime]::UtcNow.ToString('o')
                drivers = @(Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion)
                source_commit = (& git rev-parse HEAD)
                source_files = @(Get-ChildItem native/examples/interop/*.rs, "native/examples/$example.rs", native/Cargo.lock | Get-FileHash -Algorithm SHA256 | Select-Object Path, Hash)
                core_archive_sha256 = if ($env:OIDN_WEBGPU_CORE_ARCHIVE) { (Get-FileHash -LiteralPath $env:OIDN_WEBGPU_CORE_ARCHIVE -Algorithm SHA256).Hash } else { $null }
                validation_environment = @{}; report = $null
            }
            foreach ($variable in @('VK_LAYER_PATH', 'VK_LAYER_GPUAV_ENABLE', 'VK_LAYER_VALIDATE_SYNC', 'WGPU_GPU_BASED_VALIDATION', 'WGPU_DX12_AGILITY_SDK_PATH', 'WGPU_DX12_AGILITY_SDK_VERSION')) {
                $metadata.validation_environment[$variable] = [Environment]::GetEnvironmentVariable($variable)
            }
            if (!$timedOut -and $process.ExitCode -eq 0) {
                $jsonLine = @($outText -split '\r?\n' | Where-Object { $_.StartsWith('{') })[-1]
                $metadata.report = $jsonLine | ConvertFrom-Json
                if (!$metadata.report.output_byte_equal) { throw "$name did not report byte-equal output." }
            }
            $metadata | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath (Join-Path $output.FullName "$name.metadata.json") -Encoding utf8
            $reports.Add($metadata)
            $reports | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath (Join-Path $output.FullName "$Backend-summary.json") -Encoding utf8
            $process.Dispose()
            if ($timedOut -or $metadata.exit_code -ne 0) { throw "$name failed; see its stderr log and metadata in $($output.FullName)." }
            Write-Output "$name : $Frames frames, actual precision=$($metadata.report.precision), byte equal; raw results in $($output.FullName)"
        }
    }
} finally { Pop-Location }
