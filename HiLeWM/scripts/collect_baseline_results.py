#!/usr/bin/env python3
"""Parse LeWM eval result files and export Stage-1 baseline JSON artifacts."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "stage1_baseline"
DEFAULT_STABLEWM_HOME = Path.home() / ".stable-wm"


def _gpu_info() -> dict:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if out:
            name, total, used = [x.strip() for x in out.split(",", 2)]
            return {
                "gpu_name": name,
                "gpu_memory_total_mb": int(total),
                "gpu_memory_used_mb": int(used),
            }
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        pass
    return {}


def _git_commit(repo: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=repo,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _parse_results_file(path: Path) -> list[dict]:
    text = path.read_text()
    blocks = re.split(r"\n==== CONFIG ====\n", text)
    runs: list[dict] = []
    for block in blocks[1:]:
        config_part, _, results_part = block.partition("\n==== RESULTS ====\n")
        config: dict = {}
        for line in config_part.splitlines():
            if line.startswith("seed:"):
                config["seed"] = int(line.split(":", 1)[1].strip())
            elif line.startswith("policy:"):
                config["policy"] = line.split(":", 1)[1].strip()
            elif line.startswith("  env_name:"):
                config["env_name"] = line.split(":", 1)[1].strip()
            elif line.startswith("  num_eval:"):
                config["num_eval"] = int(line.split(":", 1)[1].strip())
            elif line.startswith("  goal_offset_steps:"):
                config["goal_offset_steps"] = int(line.split(":", 1)[1].strip())
            elif line.startswith("  eval_budget:"):
                config["eval_budget"] = int(line.split(":", 1)[1].strip())
            elif line.startswith("  dataset_name:"):
                config["dataset_name"] = line.split(":", 1)[1].strip()
            elif line.startswith("  horizon:"):
                config["cem_horizon"] = int(line.split(":", 1)[1].strip())
            elif line.startswith("  num_samples:"):
                config["cem_candidates"] = int(line.split(":", 1)[1].strip())
            elif line.startswith("  n_steps:"):
                config["cem_n_steps"] = int(line.split(":", 1)[1].strip())
            elif line.startswith("  topk:"):
                config["cem_topk"] = int(line.split(":", 1)[1].strip())

        sr_match = re.search(r"'success_rate':\s*([0-9.]+)", results_part)
        time_match = re.search(
            r"evaluation_time:\s*([0-9.]+)\s*seconds", results_part
        )
        if not sr_match:
            continue

        success_rate = float(sr_match.group(1))
        successes_match = re.search(
            r"'episode_successes':\s*array\(\[(.*?)\]\)",
            results_part,
            re.DOTALL,
        )
        successes = None
        if successes_match:
            raw = successes_match.group(1).replace("\n", " ")
            successes = [
                item.strip() == "True"
                for item in raw.split(",")
                if item.strip()
            ]

        wall_clock = float(time_match.group(1)) if time_match else None
        num_eval = config.get("num_eval") or len(successes or [])
        planning_time_per_step = None
        if wall_clock and config.get("eval_budget") and num_eval:
            planning_time_per_step = wall_clock / (num_eval * config["eval_budget"])

        runs.append(
            {
                "env_name": config.get("env_name"),
                "method": "LeWM+CEM",
                "checkpoint": config.get("policy"),
                "seed": config.get("seed"),
                "success_rate": success_rate,
                "num_successes": int(sum(bool(x) for x in (successes or []))),
                "num_eval": num_eval,
                "episode_successes": successes,
                "goal_offset_steps": config.get("goal_offset_steps"),
                "eval_budget": config.get("eval_budget"),
                "cem_horizon": config.get("cem_horizon"),
                "cem_candidates": config.get("cem_candidates"),
                "cem_n_steps": config.get("cem_n_steps"),
                "cem_topk": config.get("cem_topk"),
                "planning_time_per_step": planning_time_per_step,
                "wall_clock_time_sec": wall_clock,
                "dataset_name": config.get("dataset_name"),
                "results_file": str(path),
                "git_commit": _git_commit(ROOT / "le-wm"),
                "collected_at_utc": datetime.now(timezone.utc).isoformat(),
                **_gpu_info(),
            }
        )
    return runs


def _env_slug(env_name: str | None) -> str:
    if not env_name:
        return "unknown"
    return env_name.split("/")[-1].replace("-v1", "").lower()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stablewm-home",
        default=str(DEFAULT_STABLEWM_HOME),
        help="STABLEWM_HOME root (default: ~/.stable-wm)",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT),
        help="Output directory for per-run JSON files",
    )
    args = parser.parse_args()

    stablewm_home = Path(args.stablewm_home).expanduser()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result_files = sorted((stablewm_home / "quentinll").glob("*_results.txt"))
    if not result_files:
        raise SystemExit(f"No *_results.txt under {stablewm_home / 'quentinll'}")

    all_runs: list[dict] = []
    for path in result_files:
        for run in _parse_results_file(path):
            slug = _env_slug(run["env_name"])
            seed = run.get("seed", "unknown")
            out_path = out_dir / f"baseline_{slug}_seed{seed}.json"
            out_path.write_text(json.dumps(run, indent=2))
            all_runs.append(run)
            print(f"wrote {out_path}")

    summary: dict = {}
    for run in all_runs:
        slug = _env_slug(run["env_name"])
        summary.setdefault(slug, []).append(run["success_rate"])

    summary_stats = {
        env: {
            "n_runs": len(rates),
            "mean_success_rate": round(sum(rates) / len(rates), 2),
            "min_success_rate": min(rates),
            "max_success_rate": max(rates),
            "seeds": [
                r["seed"]
                for r in all_runs
                if _env_slug(r["env_name"]) == env
            ],
        }
        for env, rates in summary.items()
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "stage": "stage1_baseline",
                "collected_at_utc": datetime.now(timezone.utc).isoformat(),
                "environments": summary_stats,
                "runs": all_runs,
            },
            indent=2,
        )
    )
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
