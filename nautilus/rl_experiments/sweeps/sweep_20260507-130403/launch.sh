#!/bin/bash -l
#
#SBATCH --array=0-0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --job-name=sweep_20260507-130403
#SBATCH --output=%x_%A_%a.out
# stdout/stderr -> sweep_20260507-130403_<jobid>_<arrayidx>.out in $SLURM_SUBMIT_DIR.

# Sweep id : sweep_20260507-130403
# Trials   : 1 (Cartesian product of task=Isaac-Factory-PegInsert-Direct-v0,algorithm=ppo,total_timesteps=10000000,wandb=isaac_final)
#
# Adapted for FAU NHR Alex (AlmaLinux 8 / glibc 2.28). Bare-metal isaacsim
# wheels require glibc 2.34+, so we run the trial inside an apptainer image
# (built once from python:3.11-slim) that has torch + isaacsim 5.1 + IsaacLab
# editable pre-installed.

set -e

REPO_ROOT="$HOME/code/agentic/IsaacLab"
SIF="$REPO_ROOT/nautilus/apptainer/isaaclab.sif"

if [ ! -f "$SIF" ]; then
    echo "[ERROR] $SIF missing -- build it once with:" >&2
    echo "  cd $REPO_ROOT/nautilus/apptainer && apptainer build --fakeroot --force isaaclab.sif isaaclab.def" >&2
    exit 1
fi

# WANDB credentials live in $HOME/.netrc on the host; apptainer auto-mounts
# $HOME so they're available inside the container. Override only if the env
# var is set in the SLURM environment.
export WANDB_API_KEY=${WANDB_API_KEY:-}

# --- Trial table (parallel arrays; index = $SLURM_ARRAY_TASK_ID) ---
TRIAL_IDS=(
  "000_ppo_Isaac-Factory-PegInsert-Direct-v0_827200d1"
)
TRIAL_CMDS=(
  "python nautilus/scripts/rl/custom_torch/train.py --config-name=ppo.parallel task=Isaac-Factory-PegInsert-Direct-v0 total_timesteps=10000000 wandb=isaac_final"
)

ID="${TRIAL_IDS[$SLURM_ARRAY_TASK_ID]}"
CMD="${TRIAL_CMDS[$SLURM_ARRAY_TASK_ID]}"
echo "[sweep sweep_20260507-130403] trial $SLURM_ARRAY_TASK_ID :: $ID"
echo "[host] $(hostname); GPU(s):"
nvidia-smi -L 2>&1 | head -5
echo "[cmd] apptainer exec --nv --bind $REPO_ROOT:/repo/IsaacLab --pwd /repo/IsaacLab $SIF $CMD"

# OMNI_KIT_ACCEPT_EULA=YES + PRIVACYACCEPT=YES skip the interactive Omniverse
# EULA / privacy prompts on first-run; required for non-interactive SLURM jobs.
# --writable-tmpfs gives Kit a RAM-backed overlay so its writes to
# /opt/venv/.../isaacsim/kit/{cache,data} succeed (the .sif is read-only).
apptainer exec \
    --nv \
    --writable-tmpfs \
    --bind "$REPO_ROOT:/repo/IsaacLab" \
    --pwd /repo/IsaacLab \
    --env OMNI_KIT_ACCEPT_EULA=YES \
    --env PRIVACYACCEPT=YES \
    --env WANDB_API_KEY="$WANDB_API_KEY" \
    "$SIF" \
    bash -c "$CMD"
