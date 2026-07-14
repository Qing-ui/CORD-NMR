[CmdletBinding()]
param(
    [string]$RuntimeRoot,
    [string]$MicromambaRoot,
    [string]$EnvironmentRoot,
    [string]$AssetSourceDirectory,
    [switch]$Force,
    [switch]$SkipNMRNet,
    [switch]$SkipCascade2
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$releaseTag = "v1.0.0"
$assetBaseUrl = "https://github.com/Qing-ui/CORD-NMR/releases/download/$releaseTag"
$uniCoreRevision = "ace6fae1c8479a9751f2bb1e1d6e4047427bc134"
$micromambaVersion = "2.8.1-0"

$assets = @{
    "nmrnet-liquid-C-checkpoint_best.pt" = "0EFB06C233A6E851629F695E516D94209963FB60DF39194A5DBB20BD3CB9DB7B"
    "nmrnet-liquid-H-checkpoint_best.pt" = "BB25D02A103A63C54CB4DBD0A010BF5425FFB3C15BA8209CC38BAB12F0A2341F"
    "nmrnet-liquid-metadata.zip" = "E809D46E253371E663F513667EB87F5401A7014B59ED443B612772DFD3DF0302"
    "cascade2-predict-smiles-ff-gpr.zip" = "D8399C21681F4195966A6723A1E50E0B4DBA0C1CB50C6F8580AC92C33926760C"
}

$micromambaAsset = "micromamba-win-64.exe"
$micromambaUrl = "https://github.com/mamba-org/micromamba-releases/releases/download/$micromambaVersion/$micromambaAsset"
$micromambaSha256 = "8A51F88EC02600488EA20C3ACD93FBD4DA6C0F03FC499AA53FD234C6749B94B0"
$uniCoreAsset = "Uni-Core-$($uniCoreRevision.Substring(0, 8)).zip"
$uniCoreUrl = "https://github.com/dptech-corp/Uni-Core/archive/$uniCoreRevision.zip"
$uniCoreSha256 = "D295489B0C85253E3C1F979ACBBE04F6BBCD9DC99C1FA2638A5D5EF9EDB3DEE3"

$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $RuntimeRoot) {
    $RuntimeRoot = Join-Path $projectRoot "external\NMR-Predictor-Portable"
}
if (-not $MicromambaRoot) {
    $MicromambaRoot = Join-Path $env:LOCALAPPDATA "CORD-NMR\micromamba"
}
if (-not $EnvironmentRoot) {
    $EnvironmentRoot = Join-Path $env:LOCALAPPDATA "CORD-NMR\envs"
}

$RuntimeRoot = [IO.Path]::GetFullPath($RuntimeRoot)
$MicromambaRoot = [IO.Path]::GetFullPath($MicromambaRoot)
$EnvironmentRoot = [IO.Path]::GetFullPath($EnvironmentRoot)
$cacheRoot = Join-Path $RuntimeRoot "cache\installer"
$envRoot = $EnvironmentRoot
$modelRoot = Join-Path $RuntimeRoot "models"
$licenseRoot = Join-Path $RuntimeRoot "licenses"
$appRoot = Join-Path $RuntimeRoot "app"

function Write-Step([string]$Message) {
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

function Assert-LastExitCode([string]$Action) {
    if ($LASTEXITCODE -ne 0) {
        throw "$Action failed with exit code $LASTEXITCODE."
    }
}

function Test-FileSha256([string]$Path, [string]$Expected) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $false
    }
    $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
    return $actual.Equals($Expected, [StringComparison]::OrdinalIgnoreCase)
}

function Get-VerifiedFile {
    param(
        [string]$Name,
        [string]$Url,
        [string]$Sha256
    )

    $destination = Join-Path $cacheRoot $Name
    if ((-not $Force) -and (Test-FileSha256 $destination $Sha256)) {
        Write-Host "Using cached $Name"
        return $destination
    }

    $partial = "$destination.partial"
    if (Test-Path -LiteralPath $partial -PathType Leaf) {
        Remove-Item -LiteralPath $partial -Force
    }

    if ($AssetSourceDirectory) {
        $source = Join-Path ([IO.Path]::GetFullPath($AssetSourceDirectory)) $Name
        if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
            throw "Local asset is missing: $source"
        }
        Copy-Item -LiteralPath $source -Destination $partial -Force
    } else {
        Write-Host "Downloading $Name"
        $downloaded = $false
        if (Get-Command Start-BitsTransfer -ErrorAction SilentlyContinue) {
            try {
                Start-BitsTransfer -Source $Url -Destination $partial -DisplayName "CORD-NMR $Name"
                $downloaded = $true
            } catch {
                Write-Host "BITS download unavailable; using HTTPS fallback."
            }
        }
        if (-not $downloaded) {
            Invoke-WebRequest -Uri $Url -OutFile $partial -Headers @{ "User-Agent" = "CORD-NMR-installer" } -UseBasicParsing
        }
    }

    if (-not (Test-FileSha256 $partial $Sha256)) {
        throw "SHA-256 verification failed for $Name."
    }
    Move-Item -LiteralPath $partial -Destination $destination -Force
    return $destination
}

function Invoke-Python {
    param(
        [string]$Python,
        [string[]]$Arguments,
        [string]$Action,
        [string]$WorkingDirectory
    )
    if ($WorkingDirectory) {
        Push-Location -LiteralPath $WorkingDirectory
    }
    try {
        & $Python @Arguments
        Assert-LastExitCode $Action
    } finally {
        if ($WorkingDirectory) {
            Pop-Location
        }
    }
}

function Ensure-Micromamba {
    $micromambaExe = Join-Path $MicromambaRoot "micromamba.exe"
    if ((-not $Force) -and (Test-FileSha256 $micromambaExe $micromambaSha256)) {
        return $micromambaExe
    }

    Write-Step "Installing the private micromamba runtime"
    $download = Get-VerifiedFile -Name $micromambaAsset -Url $micromambaUrl -Sha256 $micromambaSha256
    New-Item -ItemType Directory -Path $MicromambaRoot -Force | Out-Null
    Copy-Item -LiteralPath $download -Destination $micromambaExe -Force
    if (-not (Test-FileSha256 $micromambaExe $micromambaSha256)) {
        throw "micromamba installation failed SHA-256 verification."
    }
    return $micromambaExe
}

function Ensure-Environment {
    param(
        [string]$MicromambaExe,
        [string]$Prefix,
        [string]$Requirements,
        [string]$Name
    )

    $python = Join-Path $Prefix "python.exe"
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        Write-Step "Creating the $Name environment"
        $mambaRootPrefix = Join-Path $MicromambaRoot "root"
        & $MicromambaExe --root-prefix $mambaRootPrefix create --yes --prefix $Prefix --override-channels --channel "https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge" python=3.10.20 pip=26.0.1 | Out-Host
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "The primary conda-forge mirror failed; retrying with the official endpoint."
            & $MicromambaExe --root-prefix $mambaRootPrefix create --yes --prefix $Prefix --override-channels --channel conda-forge python=3.10.20 pip=26.0.1 | Out-Host
        }
        Assert-LastExitCode "Creating the $Name environment"
    }

    Write-Step "Installing pinned $Name dependencies"
    $pipOptions = @("--quiet", "--disable-pip-version-check", "--no-warn-script-location", "--timeout", "30", "--retries", "2")
    & $python -m pip install @pipOptions --requirement $Requirements | Out-Host
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "The primary Python package index failed; retrying with the Tsinghua mirror."
        & $python -m pip install @pipOptions --index-url "https://pypi.tuna.tsinghua.edu.cn/simple" --requirement $Requirements | Out-Host
    }
    Assert-LastExitCode "Installing $Name dependencies"
    if ($Name -eq "CASCADE-2.0") {
        & $python -m pip install @pipOptions --no-deps kgcnn==2.2.1 | Out-Host
        if ($LASTEXITCODE -ne 0) {
            & $python -m pip install @pipOptions --no-deps --index-url "https://pypi.tuna.tsinghua.edu.cn/simple" kgcnn==2.2.1 | Out-Host
        }
        Assert-LastExitCode "Installing the CASCADE-2.0 KGCNN runtime"
    }
    return $python
}

function Install-UniCore {
    Write-Step "Installing the pinned Uni-Core Python runtime"
    $archive = Get-VerifiedFile -Name $uniCoreAsset -Url $uniCoreUrl -Sha256 $uniCoreSha256
    $extractRoot = Join-Path $cacheRoot "Uni-Core-$uniCoreRevision"
    if (-not (Test-Path -LiteralPath $extractRoot -PathType Container)) {
        Expand-Archive -LiteralPath $archive -DestinationPath $extractRoot -Force
    }
    $sourceRoot = Get-ChildItem -LiteralPath $extractRoot -Directory | Select-Object -First 1
    if (-not $sourceRoot) {
        throw "Uni-Core source archive did not contain a root directory."
    }
    Copy-Item -LiteralPath (Join-Path $sourceRoot.FullName "unicore") -Destination $appRoot -Recurse -Force
    Copy-Item -LiteralPath (Join-Path $sourceRoot.FullName "LICENSE") -Destination (Join-Path $licenseRoot "LICENSE-Uni-Core.txt") -Force
}

function Install-NMRNetModels {
    Write-Step "Installing the NMRNet liquid-state model files"
    $metadata = Get-VerifiedFile -Name "nmrnet-liquid-metadata.zip" -Url "$assetBaseUrl/nmrnet-liquid-metadata.zip" -Sha256 $assets["nmrnet-liquid-metadata.zip"]
    $cWeight = Get-VerifiedFile -Name "nmrnet-liquid-C-checkpoint_best.pt" -Url "$assetBaseUrl/nmrnet-liquid-C-checkpoint_best.pt" -Sha256 $assets["nmrnet-liquid-C-checkpoint_best.pt"]
    $hWeight = Get-VerifiedFile -Name "nmrnet-liquid-H-checkpoint_best.pt" -Url "$assetBaseUrl/nmrnet-liquid-H-checkpoint_best.pt" -Sha256 $assets["nmrnet-liquid-H-checkpoint_best.pt"]

    $nmrnetRoot = Join-Path $modelRoot "nmrnet"
    Expand-Archive -LiteralPath $metadata -DestinationPath $nmrnetRoot -Force
    $cDirectory = Join-Path $nmrnetRoot "liquid\C_mol_pre_all_h_220816_global_0_kener_gauss_atomdes_0_unimol_large_atom_regloss_mae_lr_1e-3_bs_16_0.06_200\cv_seed_42_fold_0"
    $hDirectory = Join-Path $nmrnetRoot "liquid\H_mol_pre_all_h_220816_global_0_kener_gauss_atomdes_0_unimol_large_atom_regloss_mae_lr_5e-3_bs_16_0.06_400\cv_seed_42_fold_0"
    New-Item -ItemType Directory -Path $cDirectory, $hDirectory -Force | Out-Null
    Copy-Item -LiteralPath $cWeight -Destination (Join-Path $cDirectory "checkpoint_best.pt") -Force
    Copy-Item -LiteralPath $hWeight -Destination (Join-Path $hDirectory "checkpoint_best.pt") -Force
    Copy-Item -LiteralPath (Join-Path $nmrnetRoot "LICENSE-NMRNet.txt") -Destination $licenseRoot -Force
}

function Install-Cascade2Model {
    Write-Step "Installing the CASCADE-2.0 inference model"
    $archive = Get-VerifiedFile -Name "cascade2-predict-smiles-ff-gpr.zip" -Url "$assetBaseUrl/cascade2-predict-smiles-ff-gpr.zip" -Sha256 $assets["cascade2-predict-smiles-ff-gpr.zip"]
    $cascadeRoot = Join-Path $modelRoot "cascade2"
    Expand-Archive -LiteralPath $archive -DestinationPath $cascadeRoot -Force
    Copy-Item -LiteralPath (Join-Path $cascadeRoot "LICENSE-CASCADE-2.0.txt") -Destination $licenseRoot -Force
}

function Test-InstalledRuntime {
    param(
        [string]$ApplicationPython,
        [string]$NMRNetPython,
        [string]$Cascade2Python
    )

    Write-Step "Verifying the installed CORD-NMR runtime"
    $appCheck = "import tkinter; import numpy, pandas, scipy, sklearn, matplotlib, PIL, rdkit, tqdm, openpyxl; import gui; print('CORD-NMR GUI runtime OK')"
    Invoke-Python -Python $ApplicationPython -Arguments @("-c", $appCheck) -Action "Loading the CORD-NMR GUI runtime" -WorkingDirectory $projectRoot
    if ($NMRNetPython) {
        $nmrScript = Join-Path $appRoot "script\predict_liquid_batch.py"
        Invoke-Python -Python $NMRNetPython -Arguments @($nmrScript, "--help") -Action "Loading the NMRNet runtime"
    }
    if ($Cascade2Python) {
        $cascadeModel = Join-Path $modelRoot "cascade2\Predict_SMILES_FF_GPR"
        $verify = "import os, sys; os.chdir(r'$cascadeModel'); sys.path.insert(0, r'$cascadeModel'); sys.path.insert(0, r'$cascadeModel\modules'); import tensorflow as tf; import tensorflow_probability as tfp; from nfp.preprocessing import GraphSequence; from model import make_model; model=make_model(); model.load_weights(r'$cascadeModel\best_model_val_mae.h5'); print('CASCADE-2.0 runtime OK')"
        Invoke-Python -Python $Cascade2Python -Arguments @("-c", $verify) -Action "Loading the CASCADE-2.0 runtime"
    }
}

if (-not [Environment]::Is64BitOperatingSystem -or $env:OS -ne "Windows_NT") {
    throw "This installer supports 64-bit Windows only."
}

Write-Host "CORD-NMR installer" -ForegroundColor Green
Write-Host "Runtime root: $RuntimeRoot"
Write-Host "Environment root: $EnvironmentRoot"
Write-Host "Third-party licenses: $projectRoot\docs\THIRD_PARTY_NOTICES.md"

New-Item -ItemType Directory -Path $RuntimeRoot, $cacheRoot, $envRoot, $modelRoot, $licenseRoot -Force | Out-Null
$environmentCacheRoot = Join-Path $EnvironmentRoot ".installer-cache"
$tempRoot = Join-Path $environmentCacheRoot "temp"
$pipCacheRoot = Join-Path $environmentCacheRoot "pip"
New-Item -ItemType Directory -Path $tempRoot, $pipCacheRoot -Force | Out-Null
$env:TEMP = $tempRoot
$env:TMP = $tempRoot
$env:PIP_CACHE_DIR = $pipCacheRoot
$micromambaExe = Ensure-Micromamba
$applicationPython = Ensure-Environment -MicromambaExe $micromambaExe -Prefix (Join-Path $envRoot "app") -Requirements (Join-Path $projectRoot "requirements.txt") -Name "CORD-NMR GUI"
$nmrnetPython = $null
$cascade2Python = $null

if (-not $SkipNMRNet) {
    $nmrnetPython = Ensure-Environment -MicromambaExe $micromambaExe -Prefix (Join-Path $envRoot "nmrnet") -Requirements (Join-Path $RuntimeRoot "requirements-nmrnet.txt") -Name "NMRNet"
    Install-UniCore
    Install-NMRNetModels
}

if (-not $SkipCascade2) {
    $cascade2Python = Ensure-Environment -MicromambaExe $micromambaExe -Prefix (Join-Path $envRoot "cascade2") -Requirements (Join-Path $RuntimeRoot "requirements-cascade2.txt") -Name "CASCADE-2.0"
    Install-Cascade2Model
}

Copy-Item -LiteralPath (Join-Path $projectRoot "docs\THIRD_PARTY_NOTICES.md") -Destination (Join-Path $licenseRoot "THIRD_PARTY_NOTICES.md") -Force
Test-InstalledRuntime -ApplicationPython $applicationPython -NMRNetPython $nmrnetPython -Cascade2Python $cascade2Python

$runtimePathsFile = Join-Path $RuntimeRoot "runtime-paths.json"
$existingRuntimePaths = @{}
if (Test-Path -LiteralPath $runtimePathsFile -PathType Leaf) {
    try {
        $existingRuntimePaths = Get-Content -LiteralPath $runtimePathsFile -Raw -Encoding utf8 | ConvertFrom-Json
    } catch {
        $existingRuntimePaths = @{}
    }
}
$runtimePaths = [ordered]@{
    application_python = $applicationPython
    nmrnet_python = if ($nmrnetPython) { $nmrnetPython } else { $existingRuntimePaths.nmrnet_python }
    cascade2_python = if ($cascade2Python) { $cascade2Python } else { $existingRuntimePaths.cascade2_python }
}
$runtimePaths | ConvertTo-Json | Set-Content -LiteralPath $runtimePathsFile -Encoding utf8

$state = [ordered]@{
    installed_at = (Get-Date).ToString("o")
    release_tag = $releaseTag
    runtime_root = $RuntimeRoot
    environment_root = $EnvironmentRoot
    application = [bool]$applicationPython
    nmrnet = [bool]$nmrnetPython
    cascade2 = [bool]$cascade2Python
    unicore_revision = $uniCoreRevision
    asset_sha256 = $assets
}
$state | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $RuntimeRoot "install-state.json") -Encoding utf8

Write-Host "`nCORD-NMR is installed and ready." -ForegroundColor Green
Write-Host "Start the application with run_gui.bat."
