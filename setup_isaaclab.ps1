<#
.SYNOPSIS
    Installs Isaac Lab for this project (Isaac Sim 6.0 standalone, Windows) in one step.

.DESCRIPTION
    Safe to re-run: every step checks what is already done and skips it.

      1. Checks git, the NVIDIA driver, the Isaac Sim folder and Windows long-path support.
      2. Clones Isaac Lab at the release that matches Isaac Sim 6.0 (v3.0.0-beta2.patch1).
      3. Links the Isaac Sim folder into it as _isaac_sim (a directory junction; no admin rights needed).
      4. Applies the Franka model path fix (the model moved to a Legacy/ folder on NVIDIA's asset server).
      5. Runs Isaac Lab's installer, which installs into Isaac Sim's bundled Python (about 9 GB, 10-30 minutes).
      6. Verifies the install with a short headless run of this project's isaaclab_pallet_cells.py.

    See docs/isaac_lab_setup.md for what each step does and why.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\setup_isaaclab.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\setup_isaaclab.ps1 -IsaacSimPath D:\isaacsim -IsaacLabPath D:\IsaacLab
#>

[CmdletBinding()]
param(
    # Isaac Sim 6.0 standalone folder (the one that contains python.bat).
    [string]$IsaacSimPath = (Join-Path $HOME "Downloads\isaac-sim-standalone-6.0.0-windows-x86_64"),
    # Where to put Isaac Lab. Keep it short to stay clear of Windows path-length limits.
    [string]$IsaacLabPath = (Join-Path $HOME "IsaacLab"),
    # Isaac Lab release for Isaac Sim 6.0.
    [string]$Tag = "v3.0.0-beta2.patch1",
    # Skip the installer (step 5), e.g. when Isaac Lab is already installed.
    [switch]$SkipInstall,
    # Skip the verification run (step 6).
    [switch]$SkipVerify
)

$ErrorActionPreference = "Stop"
$RepoUrl = "https://github.com/isaac-sim/IsaacLab.git"
$ProjectDir = $PSScriptRoot

function Step($n, $text) { Write-Host ""; Write-Host "[$n/6] $text" -ForegroundColor Cyan }
function Ok($text) { Write-Host "      OK   $text" -ForegroundColor Green }
function Skip($text) { Write-Host "      SKIP $text" -ForegroundColor DarkGray }
function Warn($text) { Write-Host "      WARN $text" -ForegroundColor Yellow }
function Fail($text) { Write-Host "      FAIL $text" -ForegroundColor Red; exit 1 }

# --- 1. prerequisites ----------------------------------------------------------------------------------------------
Step 1 "Checking prerequisites"
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { Fail "git is not installed (https://git-scm.com/download/win)." }
Ok "git found"

$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($smi) {
    $gpu = (& nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>$null | Select-Object -First 1)
    Ok "NVIDIA GPU: $gpu"
} else {
    Warn "nvidia-smi not found: Isaac Sim needs an NVIDIA RTX GPU and driver."
}

if (-not (Test-Path (Join-Path $IsaacSimPath "python.bat"))) {
    Fail "Isaac Sim not found at $IsaacSimPath (no python.bat). Pass -IsaacSimPath <folder>."
}
Ok "Isaac Sim: $IsaacSimPath"

$longPaths = (Get-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -ErrorAction SilentlyContinue).LongPathsEnabled
if ($longPaths -eq 1) {
    Ok "Windows long paths enabled"
} else {
    Warn "Windows long paths are off. If the install fails on long file names, run this once in an administrator PowerShell:"
    Write-Host '           New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force'
}

# --- 2. clone ------------------------------------------------------------------------------------------------------
Step 2 "Getting Isaac Lab $Tag"
if (Test-Path (Join-Path $IsaacLabPath ".git")) {
    $current = (& git -C $IsaacLabPath describe --tags --exact-match 2>$null)
    if ($current -eq $Tag) {
        Skip "already cloned at $IsaacLabPath ($Tag)"
    } else {
        Warn "$IsaacLabPath is a git repo at '$current', not $Tag. Leaving it as is; delete it or pass -IsaacLabPath to use a fresh folder."
    }
} elseif (Test-Path $IsaacLabPath) {
    Fail "$IsaacLabPath exists but is not a git repository. Remove it or pass -IsaacLabPath <new folder>."
} else {
    # Git for Windows ignores the Windows long-path setting unless core.longpaths is on, and Isaac Lab has some
    # very deep file paths.
    & git -c core.longpaths=true clone --depth 1 --branch $Tag $RepoUrl $IsaacLabPath
    if ($LASTEXITCODE -ne 0) { Fail "git clone failed (see above). Delete $IsaacLabPath before re-running." }
    & git -C $IsaacLabPath config core.longpaths true
    Ok "cloned to $IsaacLabPath"
}
# A clone that stopped partway (for example on a too-long file name) leaves files missing: catch that.
$missing = @(& git -C $IsaacLabPath -c core.longpaths=true status --porcelain 2>$null |
    Where-Object { $_ -match '^\s*D ' })
if ($missing.Count -gt 0) {
    Fail "$($missing.Count) files are missing from $IsaacLabPath (the checkout did not finish). Delete the folder and re-run, or pass a shorter -IsaacLabPath."
}
Ok "all files present"

# --- 3. link Isaac Sim ---------------------------------------------------------------------------------------------
Step 3 "Linking Isaac Sim into Isaac Lab (_isaac_sim)"
$link = Join-Path $IsaacLabPath "_isaac_sim"
if (Test-Path $link) {
    $item = Get-Item $link -Force
    $target = @($item.Target)[0]
    if ($item.LinkType -and $target -and ((Resolve-Path $target).Path -eq (Resolve-Path $IsaacSimPath).Path)) {
        Skip "_isaac_sim already points to $IsaacSimPath"
    } else {
        Warn "_isaac_sim exists but points to '$target'. Leaving it as is."
    }
} else {
    New-Item -ItemType Junction -Path $link -Target $IsaacSimPath | Out-Null
    Ok "_isaac_sim -> $IsaacSimPath"
}

# --- 4. Franka model path fix -------------------------------------------------------------------------------------
Step 4 "Applying the Franka model path fix"
$franka = Join-Path $IsaacLabPath "source\isaaclab_assets\isaaclab_assets\robots\franka.py"
$old = '{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd'
$new = '{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/Legacy/panda_instanceable.usd'
if (-not (Test-Path $franka)) {
    Warn "franka.py not found; skipping (only Isaac Lab's own Franka tasks need this fix)."
} else {
    $text = [System.IO.File]::ReadAllText($franka)
    if ($text.Contains($new)) {
        Skip "already applied"
    } elseif ($text.Contains($old)) {
        [System.IO.File]::WriteAllText($franka, $text.Replace($old, $new), (New-Object System.Text.UTF8Encoding($false)))
        Ok "franka.py now points to Legacy/panda_instanceable.usd"
    } else {
        Warn "franka.py has neither the old nor the new path; Isaac Lab may have changed. Check it by hand."
    }
}

# --- 5. install ----------------------------------------------------------------------------------------------------
Step 5 "Installing Isaac Lab into Isaac Sim's Python"
$bat = Join-Path $IsaacLabPath "isaaclab.bat"
if ($SkipInstall) {
    Skip "-SkipInstall given"
} else {
    Write-Host "      Running isaaclab.bat --install (about 9 GB of downloads, 10-30 minutes)..."
    Push-Location $IsaacLabPath
    try {
        & cmd /c "`"$bat`" --install"
        $code = $LASTEXITCODE
    } finally {
        Pop-Location
    }
    if ($code -ne 0) { Fail "isaaclab.bat --install exited with code $code." }
    Ok "installed (pip may print version-conflict warnings between RL libraries; they are expected)"
}

# --- 6. verify -----------------------------------------------------------------------------------------------------
Step 6 "Verifying with this project's Isaac Lab cells (headless, 1 cell, 1 box)"
$cells = Join-Path $ProjectDir "isaaclab_pallet_cells.py"
if ($SkipVerify) {
    Skip "-SkipVerify given"
} elseif (-not (Test-Path $cells)) {
    Warn "isaaclab_pallet_cells.py not found next to this script; skipping."
} else {
    Write-Host "      First launch compiles shaders and downloads assets; this can take several minutes."
    Push-Location $IsaacLabPath
    try {
        $out = & cmd /c "`"$bat`" -p `"$cells`" --num_cells 1 --max_boxes 1 --test --no_learning --env grid 2>&1"
    } finally {
        Pop-Location
    }
    $placed = $out | Select-String -Pattern "\[ep 1\].*placed"
    if ($placed) {
        Ok ($placed[0].Line.Trim())
    } else {
        $out | Select-Object -Last 25 | ForEach-Object { Write-Host "      $_" }
        Fail "The verification run did not place a box (output above)."
    }
    $log = Join-Path $ProjectDir "training_log_isaaclab.csv"
    if (Test-Path $log) { Remove-Item $log -ErrorAction SilentlyContinue }
}

Write-Host ""
Write-Host "Isaac Lab is ready. To watch the four warehouse cells, from a Command Prompt:" -ForegroundColor Green
Write-Host "    cd /d `"$IsaacLabPath`""
Write-Host "    isaaclab.bat -p `"$cells`" --viz kit --speed 4"
