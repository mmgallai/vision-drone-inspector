# Depth Anything 3 integration

Used by **mapping-mode** reconstruction — when the drone sweeps the whole
room rather than orbiting a single target, the resulting captures are
poor input for splatfacto (no orbit baseline, weak SfM-like priors).
Depth Anything 3 jointly estimates depth + camera poses from arbitrary
views and produces a clean point cloud, which is what we want for
whole-scene maps.

The original splatfacto recon is unchanged for **inspection-mode** runs;
see [velocity_controller_plan.md](velocity_controller_plan.md) and the
main image_capture pipeline.

## Install (one-time)

DA3 is opt-in. The package isn't tiny (~5 GB with weights), so we don't
auto-install it.

```bash
# 1. Clone the official repo somewhere outside the workspace
cd ~/
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
cd Depth-Anything-3

# 2. Install the Python package + CLI
pip install --user xformers "torch>=2" torchvision
pip install --user -e .

# 3. (optional) Verify
da3 --help
```

Hardware: DA3 needs CUDA. Streaming inference for long videos works in
~12 GB of VRAM; image-directory mode is even lighter. Our X3-mounted
camera produces 75-150 PNGs per mapping run (≪ what DA3 streams), so
even a 8 GB laptop GPU is fine.

## How drone_recon uses it

After a mapping-mode mission ends, `image_capture._run_reconstruction`
detects the chosen `recon_method` and dispatches:

* `splatfacto` (default for `mission_mode=inspection`) — runs the
  original 5-step nerfstudio pipeline.
* `da3`        (default for `mission_mode=mapping`)    — runs
  `drone_recon.run_recon_da3`, which calls `da3 auto <captures_dir>
  --export-format ply --export-dir ~/recon_output/exports`.

Output: a single colored PLY at
`~/recon_output/exports/scene_da3.ply`. Drop it into CloudCompare,
MeshLab, Open3D, or any web 3D viewer.

You can override the default mapping at launch:

```bash
# Use DA3 even on an inspection orbit (advanced)
ros2 launch drone_recon scene1.launch.py recon_method:=da3

# Force splatfacto on a mapping sweep (will probably look bad)
ros2 launch drone_recon scene1.launch.py mission_mode:=mapping recon_method:=splatfacto
```

## Manual run on existing captures

Skip the auto-pipeline (`auto_recon:=false`) and run later:

```bash
ros2 run drone_recon run_recon_da3 ~/recon_output
```

Or directly:

```bash
da3 auto ~/recon_output/images --export-format ply \
    --export-dir ~/recon_output/exports
```

## What if DA3 isn't installed?

The dispatcher logs a clear error:

```
[recon] depth_anything_3 not on PATH. Install per
        docs/depth_anything_3.md, or pass recon_method:=splatfacto.
```

… and exits the recon thread without crashing the launch. Captures and
JSONs are already saved by `image_capture` — you can install DA3 later
and run the `run_recon_da3` CLI against the saved capture directory.
