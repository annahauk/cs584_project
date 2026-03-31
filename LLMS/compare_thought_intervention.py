"""
Run comparative evaluation for no-thought-intervention vs thought-intervention modes.

This script calls LLMS/eval_ollama_baseline.py for each requested intervention mode,
collects the generated metrics.csv files, and writes consolidated comparison outputs.

Usage example
-------------
python LLMS/compare_thought_intervention.py \
  --models gemma3:12b qwen3:8b \
  --modes none suppress_cot counterfactual_check \
  --max_samples 200

"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline vs thought-intervention modes for Ollama models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--data_dir", type=str, default="Data")
    parser.add_argument("--train_file", type=str, default="all_logic_train.csv")
    parser.add_argument("--dev_file", type=str, default="all_logic_dev.csv")
    parser.add_argument("--test_file", type=str, default="all_logic_test.csv")

    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Ollama model tags, e.g. gemma3:12b qwen3:8b",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["none", "suppress_cot", "induce_cot", "counterfactual_check", "label_then_reason"],
        help="Intervention modes to evaluate. Include 'none' for baseline comparison.",
    )
    parser.add_argument("--output_dir", type=str, default="Results")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--ollama_host", type=str, default="")
    parser.add_argument(
        "--intervention_text",
        type=str,
        default="",
        help="Optional custom intervention text for all non-'none' modes.",
    )

    return parser.parse_args()


def model_tag(models: list[str]) -> str:
    return "+".join(m.replace("/", "_").replace(":", "_") for m in models)


def data_tag(train_file: str) -> str:
    return os.path.splitext(train_file)[0].replace("_train", "")


def sample_tag(max_samples: int) -> str:
    return f"n{max_samples}" if max_samples > 0 else "nall"


def exp_prefix(train_file: str, models: list[str], max_samples: int) -> str:
    return f"data-{data_tag(train_file)}__ollama-{model_tag(models)}__{sample_tag(max_samples)}__"


def find_new_experiment_dir(output_dir: Path, prefix: str, before_dirs: set[str]) -> Path:
    all_dirs = {p.name for p in output_dir.iterdir() if p.is_dir()}
    new_dirs = sorted(d for d in (all_dirs - before_dirs) if d.startswith(prefix))
    if new_dirs:
        return output_dir / new_dirs[-1]

    candidates = sorted(
        (p for p in output_dir.iterdir() if p.is_dir() and p.name.startswith(prefix)),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No experiment directory found with prefix: {prefix}")

    return candidates[-1]


def run_one_mode(args: argparse.Namespace, mode: str, eval_script: Path) -> Path:
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    before_dirs = {p.name for p in out_root.iterdir() if p.is_dir()}

    cmd = [
        sys.executable,
        str(eval_script),
        "--data_dir",
        args.data_dir,
        "--train_file",
        args.train_file,
        "--dev_file",
        args.dev_file,
        "--test_file",
        args.test_file,
        "--models",
        *args.models,
        "--output_dir",
        args.output_dir,
        "--max_samples",
        str(args.max_samples),
        "--temperature",
        str(args.temperature),
        "--intervention",
        mode,
    ]

    if args.ollama_host:
        cmd.extend(["--ollama_host", args.ollama_host])

    if mode != "none" and args.intervention_text.strip():
        cmd.extend(["--intervention_text", args.intervention_text.strip()])

    print("\n=== Running mode:", mode, "===")
    subprocess.run(cmd, check=True)

    prefix = exp_prefix(args.train_file, args.models, args.max_samples)
    exp_dir = find_new_experiment_dir(out_root, prefix, before_dirs)

    metrics_path = exp_dir / "metrics.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Expected metrics file not found: {metrics_path}")

    return metrics_path


def build_delta_table(all_metrics: pd.DataFrame) -> pd.DataFrame:
    baseline = all_metrics[all_metrics["run_mode"] == "none"].copy()
    if baseline.empty:
        return pd.DataFrame()

    key_cols = ["model", "split"]
    metric_cols = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
        "parsed_rate",
        "seconds",
        "samples_per_sec",
    ]

    base = baseline[key_cols + metric_cols].rename(
        columns={col: f"baseline_{col}" for col in metric_cols}
    )

    merged = all_metrics.merge(base, on=key_cols, how="left")
    for col in metric_cols:
        merged[f"delta_{col}"] = merged[col] - merged[f"baseline_{col}"]

    return merged


def main() -> None:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    eval_script = script_dir / "eval_ollama_baseline.py"
    if not eval_script.is_file():
        raise FileNotFoundError(f"Could not find evaluator script: {eval_script}")

    modes = list(dict.fromkeys(args.modes))

    all_tables: list[pd.DataFrame] = []
    for mode in modes:
        metrics_path = run_one_mode(args, mode=mode, eval_script=eval_script)
        df = pd.read_csv(metrics_path)
        df["run_mode"] = mode
        all_tables.append(df)

    combined = pd.concat(all_tables, ignore_index=True)
    delta = build_delta_table(combined)

    compare_root = Path(args.output_dir) / "comparisons"
    compare_root.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"ti_compare__{data_tag(args.train_file)}__{model_tag(args.models)}__{sample_tag(args.max_samples)}__{ts}"
    run_dir = compare_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    combined_path = run_dir / "all_modes_metrics.csv"
    combined.to_csv(combined_path, index=False)

    if not delta.empty:
        delta_path = run_dir / "delta_vs_none.csv"
        delta.to_csv(delta_path, index=False)
    else:
        delta_path = None

    print("\nSaved:")
    print(f"- Combined metrics: {combined_path}")
    if delta_path is not None:
        print(f"- Delta vs none: {delta_path}")
    else:
        print("- Delta vs none: skipped (no 'none' baseline in --modes)")


if __name__ == "__main__":
    main()
