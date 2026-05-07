"""
isaaclab — single-trial driver invoked by rl-tuning-agent.

The agent decides hyperparameters and writes them to
    nautilus/rl_experiments/runs/<trial_id>/config.yaml
then calls this script to run train → eval → render in sequence.

Usage:
    python nautilus/scripts/rl/tune.py --algo ppo --task <task_id> --trial <trial_id>

This script is intentionally thin — the search loop, scoring, and
hyperparameter suggestion logic live in the agent + shared scripts under
scripts/rl-tuning-agent/.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO     = Path(__file__).resolve().parents[3]   # actual repo root
NAUTILUS = Path(__file__).resolve().parents[2]   # <repo>/nautilus


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", required=True, choices=["ppo", "sac", "td3"])
    p.add_argument("--task", required=True)
    p.add_argument("--trial", required=True)
    p.add_argument("--skip-render", action="store_true")
    return p.parse_args()


def run(cmd: list[str]) -> int:
    print(f"[tune] $ {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, cwd=str(REPO))


def main():
    args = parse_args()
    trial_dir = NAUTILUS / "rl_experiments" / "runs" / args.trial
    cfg_yaml = trial_dir / "config.yaml"
    if not cfg_yaml.exists():
        print(f"[error] expected {cfg_yaml} written by rl-tuning-agent", file=sys.stderr)
        sys.exit(2)

    train_cmd = [
        sys.executable, "nautilus/scripts/rl/train.py",
        f"--config-path={trial_dir}", "--config-name=config",
        f"algo={args.algo}", f"task={args.task}", f"trial={args.trial}",
    ]
    if run(train_cmd) != 0:
        print(f"[tune] train.py failed for trial {args.trial}", file=sys.stderr)
        sys.exit(1)

    eval_cmd = [sys.executable, "nautilus/scripts/rl/eval.py",
                "--algo", args.algo, "--task", args.task, "--trial", args.trial]
    if run(eval_cmd) != 0:
        print(f"[tune] eval.py failed for trial {args.trial}", file=sys.stderr)
        sys.exit(1)

    if not args.skip_render:
        render_cmd = [sys.executable, "nautilus/scripts/rl/render_policy.py",
                      "--algo", args.algo, "--task", args.task, "--trial", args.trial]
        # Render failure is non-fatal — log + continue so analyze/suggest can proceed.
        if run(render_cmd) != 0:
            print(f"[tune] render_policy.py failed for trial {args.trial} (non-fatal)", file=sys.stderr)


if __name__ == "__main__":
    main()
