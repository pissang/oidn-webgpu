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
    [switch]$SkipBuild,
    [switch]$Prototype,
    [switch]$RustOracle,
    [string]$VulkanIncludeDirectory = '.native-build/interop/Vulkan-Headers/include',
    [string]$Dx12IncludeDirectory = '.native-build/interop/agility/build/native/include'
)
$ErrorActionPreference = 'Stop'
if (!$IsWindows) { throw 'The current native C++ test hosts require Windows.' }
if ($Prototype -and $RustOracle) { throw '-Prototype and -RustOracle are mutually exclusive.' }
if ($Frames -lt 1 -or $Frames -gt 100000 -or $TimeoutSeconds -lt 1) { throw 'Invalid frame count or timeout.' }
if ($NoDebug -and ($Backend -ne 'dx12' -or $GpuValidation)) { throw '-NoDebug is a DX12 diagnostic mode without GPU validation.' }
$root = Split-Path $PSScriptRoot -Parent
$savedValidationEnvironment = @{}
foreach ($variable in @('VK_LAYER_GPUAV_ENABLE', 'VK_LAYER_VALIDATE_SYNC', 'WGPU_GPU_BASED_VALIDATION')) { $savedValidationEnvironment[$variable] = [Environment]::GetEnvironmentVariable($variable) }
Push-Location $root
try {
    $output = New-Item -ItemType Directory -Force $OutputDirectory
    $example = "${Backend}_interop"
    $formalDirectory = New-Item -ItemType Directory -Force '.native-build/interop/capi'
    $oracleFeatures = if ($RustOracle -and $Backend -eq 'dx12') { @('--features', 'interop-validation') } else { @() }
    if (!$SkipBuild -and ($Prototype -or $RustOracle)) {
        & cargo +stable-x86_64-pc-windows-msvc build --release --locked --target x86_64-pc-windows-msvc --manifest-path native/Cargo.toml --example $example @oracleFeatures
        if ($LASTEXITCODE -ne 0) { throw 'Cargo build failed.' }
    }
    if (!$Prototype -and !$RustOracle -and !$SkipBuild) {
        & cargo +stable-x86_64-pc-windows-msvc build --release --locked --target x86_64-pc-windows-msvc --manifest-path native/Cargo.toml --lib
        if ($LASTEXITCODE -ne 0) { throw 'DLL build failed.' }
        & cargo +stable-x86_64-pc-windows-msvc run --release --locked --target x86_64-pc-windows-msvc --manifest-path native/Cargo.toml --example interop_shaders -- "$($formalDirectory.FullName)/shaders"
        if ($LASTEXITCODE -ne 0) { throw 'Host shader build failed.' }
        $include = if ($Backend -eq 'vulkan') { $VulkanIncludeDirectory } else { $Dx12IncludeDirectory }
        $libraries = if ($Backend -eq 'dx12') { @('d3d12.lib', 'dxgi.lib', 'd3dcompiler.lib') } else { @() }
        & cl.exe /nologo /EHsc /std:c++17 /O2 /MD /Inative/include "/I$include" "native/examples/${Backend}_gpu.cpp" "/Fo$($formalDirectory.FullName)/${Backend}_gpu.obj" "/Fe$($formalDirectory.FullName)/${Backend}_gpu.exe" /link native/target/x86_64-pc-windows-msvc/release/oidn_webgpu.dll.lib @libraries
        if ($LASTEXITCODE -ne 0) { throw 'C++ DLL consumer build failed.' }
        Copy-Item native/target/x86_64-pc-windows-msvc/release/oidn_webgpu.dll $formalDirectory.FullName
    }
    $binary = if ($Prototype -or $RustOracle) { (Resolve-Path "native/target/x86_64-pc-windows-msvc/release/examples/$example.exe").Path } else { (Resolve-Path "$($formalDirectory.FullName)/${Backend}_gpu.exe").Path }
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
            if ($RustOracle) { $arguments += '--c-api' }
            if (!$Prototype -and !$RustOracle) {
                $arguments = @($case.Weights, "$($formalDirectory.FullName)/shaders", "$Frames", "$($case.Size)", "$($case.Tile)", $(if ($precision -eq 'fp32') {'1'} else {'0'}), $(if ($case.Name -eq 'base') {'1'} else {'0'}), $(if ($Unchecked) {'0'} else {'1'}))
            }
            $name = "$Backend-$($case.Name)-$precision"
            if ($GpuValidation) { $name += '-gpu-validation' }
            if ($NoDebug) { $name += '-no-debug' }
            $start = [Diagnostics.ProcessStartInfo]::new($binary)
            $start.UseShellExecute = $false
            $start.CreateNoWindow = $true
            $start.RedirectStandardOutput = $true
            $start.RedirectStandardError = $true
            $start.WorkingDirectory = $root
            $start.Environment['OIDNW_INTEROP_ADAPTER'] = $Adapter
            if ($NoDebug) { $start.Environment['OIDNW_INTEROP_NO_DEBUG'] = '1' }
            else { $start.Environment.Remove('OIDNW_INTEROP_NO_DEBUG') | Out-Null }
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
                prototype = [bool]$Prototype; rust_oracle = [bool]$RustOracle; backend = $Backend; case = $case.Name; precision = $precision
                executable = $binary; executable_sha256 = $binaryHash; arguments = $arguments
                dll_sha256 = if (!$Prototype -and !$RustOracle) { (Get-FileHash -LiteralPath "$($formalDirectory.FullName)/oidn_webgpu.dll" -Algorithm SHA256).Hash } else { $null }
                shader_files = if (!$Prototype -and !$RustOracle) { @(Get-ChildItem "$($formalDirectory.FullName)/shaders/*" | Get-FileHash -Algorithm SHA256 | Select-Object Path, Hash) } else { @() }
                weights_sha256 = (Get-FileHash -LiteralPath $case.Weights -Algorithm SHA256).Hash
                exit_code = $process.ExitCode; timed_out = $timedOut
                utc = [DateTime]::UtcNow.ToString('o')
                drivers = @(Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion)
                source_commit = (& git rev-parse HEAD)
                source_files = @(Get-ChildItem native/examples/interop/*.rs, native/examples/interop/*.hlsl, "native/examples/$example.rs", "native/examples/${Backend}_gpu.cpp", native/src/gpu_capi/*.rs, native/include/oidn_webgpu*.h, native/Cargo.lock | Get-FileHash -Algorithm SHA256 | Select-Object Path, Hash)
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
} finally {
    foreach ($variable in $savedValidationEnvironment.Keys) { [Environment]::SetEnvironmentVariable($variable, $savedValidationEnvironment[$variable], 'Process') }
    Pop-Location
}
