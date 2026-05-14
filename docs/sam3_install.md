# SAM3 weights — install / re-install

The drone_recon package needs Meta's SAM3 weights to do text-prompted
target detection. They are NOT redistributed in the repo (gated model,
~2 GB). You install them once into `~/sam3/sam3.pt`, and from then on
both the launch file and `sam3_detector` find them automatically.

The default path is `~/sam3/sam3.pt`. We deliberately keep this **outside**
the colcon build directory so a clean rebuild (`rm -rf build/ install/`)
doesn't wipe the model. The previous default was
`~/ros2_ws/build/drone_recon/sam3/sam3.pt`, which got cleaned out when
the build dir was wiped — that's why this doc exists.

## One-time setup

### 1. Get a HuggingFace account + accept SAM3 terms

`facebook/sam3` is a gated repo. You must:

1. Create a HuggingFace account at https://huggingface.co/join
2. Visit https://huggingface.co/facebook/sam3 and click **"Agree and access repository"**
3. Generate a read token at https://huggingface.co/settings/tokens
   - Click "New token", scope = "Read", name = `drone-recon` (or anything)
   - Copy the token (starts with `hf_…`)

### 2. Log in locally

```bash
huggingface-cli login
# or, with the new CLI name:
hf auth login
# Paste the token. It gets stored at ~/.cache/huggingface/token.
```

### 3. Download the weights

```bash
mkdir -p ~/sam3
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='facebook/sam3',
    local_dir='/home/mgallai/sam3',
    local_dir_use_symlinks=False,
)
"
ls -lh ~/sam3/
```

You should see the full repo contents (`sam3.pt`, `config.json`,
`model.safetensors`, tokenizer files, etc., totaling ~2 GB).

## Verifying

```bash
ls -lh ~/sam3/sam3.pt    # ~2 GB
file ~/sam3/sam3.pt      # should say "Zip archive data" (PyTorch checkpoint)
```

Then re-run the launch — `sam3_detector` will pick up the new path
automatically:

```bash
source /opt/ros/jazzy/setup.bash && source ~/ros2_ws/install/setup.bash && \
ros2 launch drone_recon scene1.launch.py
```

## Troubleshooting

### "401 Unauthorized" when downloading

Either you didn't accept the gated-repo terms (step 1.2 above), or your
HF token isn't loaded. Check with:

```bash
huggingface-cli whoami
```

It should print your username, not an error.

### Network keeps dropping mid-download

`snapshot_download` resumes from where it left off — just rerun the
Python snippet from step 3. Each chunk that already finished stays.

If your network is hostile to large downloads, you can grab the
individual files from the model page in your browser and put them in
`~/sam3/` manually.

### "SAM3 weights not found" on launch

That's the new graceful error from `sam3_detector` — it means the file
isn't at the expected path. Either:

- complete the install steps above, OR
- pass an explicit path:
  ```bash
  ros2 launch drone_recon scene1.launch.py \
      model_path:=/path/to/your/sam3.pt
  ```
- or set an env var that the launch picks up:
  ```bash
  export SAM3_WEIGHTS_PATH=/path/to/sam3.pt
  ros2 launch drone_recon scene1.launch.py
  ```

## Why this is not part of the build

SAM3 is a 2 GB gated model. We can't redistribute it inside the repo, and
we can't download it during `colcon build` (auth-gated, slow, and
flaky). Keeping the weights in `~/sam3/` makes them:

- safe from `rm -rf build/`
- shared across multiple workspace checkouts
- easy to back up
