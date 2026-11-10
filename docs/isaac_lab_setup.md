# Installing Isaac Lab for Isaac Sim 6.0 (Windows)

## Quick setup (one script)

`setup_isaaclab.ps1`, in the repo root, does every step below and then checks the result. From PowerShell, in the repo folder:

```bash
powershell -ExecutionPolicy Bypass -File .\setup_isaaclab.ps1
```

* **Defaults.** It expects Isaac Sim at `%USERPROFILE%\Downloads\isaac-sim-standalone-6.0.0-windows-x86_64` and puts Isaac Lab at `%USERPROFILE%\IsaacLab`. Use `-IsaacSimPath` and `-IsaacLabPath` to change them.
* **Re-running is safe.** Each step checks what's already done and skips it.
* **Partial runs.** `-SkipInstall` skips the roughly 9 GB install, and `-SkipVerify` skips the final test run.
* **The final check** is a headless run of `isaaclab_pallet_cells.py` with one cell and one box. It should end with a line like `[ep 1] 1/1 box #11 ... placed ... place error 0.4 mm`.

The rest of this page explains the same steps by hand.

These are the steps used for this project, with the Isaac Sim 6.0.0 standalone zip already unpacked at `C:\Users\APMayhaps\Downloads\isaac-sim-standalone-6.0.0-windows-x86_64`.

## 1. Pick the matching Isaac Lab release

Isaac Lab's `main` branch targets Isaac Sim 6.2. For **Isaac Sim 6.0**, use the tag **`v3.0.0-beta2.patch1`** (the bug-fix release of `v3.0.0-beta2`, which the Isaac Lab README lists for Isaac Sim 6.0). That release recommends Isaac Sim 6.0.1; 6.0.0 worked in testing.

## 2. Clone it to a short path

A short path avoids Windows path-length problems. Long paths were already enabled on this machine (`LongPathsEnabled = 1`).

```bash
git -c core.longpaths=true clone --depth 1 --branch v3.0.0-beta2.patch1 https://github.com/isaac-sim/IsaacLab.git C:\Users\APMayhaps\IsaacLab
```

`core.longpaths=true` matters. Git for Windows ignores the Windows long-path setting unless it's on, and in a longer folder the clone stops partway with "Filename too long" errors.

## 3. Link Isaac Sim into it

The docs use `mklink /D` from an administrator prompt. A directory junction does the same job without administrator rights. From PowerShell:

```bash
New-Item -ItemType Junction -Path C:\Users\APMayhaps\IsaacLab\_isaac_sim -Target C:\Users\APMayhaps\Downloads\isaac-sim-standalone-6.0.0-windows-x86_64
```

Isaac Lab then warns on every command that `_isaac_sim\setup_conda_env.bat` is missing. That file is only used with conda, so the warning can be ignored.

## 4. Install

From a Command Prompt:

```bash
cd /d C:\Users\APMayhaps\IsaacLab
```

```bash
isaaclab.bat --install
```

This installs into **Isaac Sim's bundled Python** (no separate environment): PyTorch 2.10 with CUDA 12.8, the Isaac Lab packages, and the RL libraries. It used about 9 GB. Pip reports a few version conflicts between the RL libraries' pins (Pillow, `packaging`, `typing_extensions`); the final versions match what Isaac Lab pins, and everything ran.

## 5. Fix the Franka model path (only for Isaac Lab's Franka tasks)

NVIDIA has since moved `panda_instanceable.usd` into a `Legacy/` folder on the Isaac Sim 6.0 asset server, so Isaac Lab's Franka tasks fail with `USD file not found`. In `C:\Users\APMayhaps\IsaacLab\source\isaaclab_assets\isaaclab_assets\robots\franka.py`, change

```
{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd
```

to

```
{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/Legacy/panda_instanceable.usd
```

Other Isaac Lab robots may have moved the same way. If a task stops with "USD file not found", look for the file in a `Legacy/` folder next to its old location.

This project's own script, `isaaclab_pallet_cells.py`, loads the UR20 and the warehouse from Isaac Sim's asset folders and doesn't need this fix.

## 6. Verify

A short headless training run with 16 Franka environments. It should print two "Learning iteration" blocks and exit:

```bash
isaaclab.bat -p scripts\reinforcement_learning\train.py --rl_library rsl_rl --task Isaac-Reach-Franka-v0 --num_envs 16 --max_iterations 2
```

Add `--viz kit` to watch. In this release `--headless` is deprecated: runs are headless unless `--viz` is given.

## Notes

* **Running from another tool.** If `isaaclab.bat` is "not recognized" even inside the IsaacLab folder, the shell has `NoDefaultCurrentDirectoryInExePath` set. Run it by its full path instead: `C:\Users\APMayhaps\IsaacLab\isaaclab.bat`.
* **Windows support.** This release's docs say Windows support is "still being completed".
* **Your other scripts.** The standalone scripts in this project still run with `python.bat` after the install; they were re-tested.
