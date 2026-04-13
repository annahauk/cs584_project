import argparse
import glob
import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Optional

import pandas as pd

"""
python run_intervention_batch.py --num_runs 5 --max_attempts 10 --cuda_visible_devices 0
"""

NUMERIC_METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1", "time_s", "train_time_s"]
OOM_ERROR_TOKENS = [
    "cuda out of memory",
    "torch.outofmemoryerror",
    "outofmemoryerror",
]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _build_batch_name(user_batch_name: Optional[str]) -> str:
    if user_batch_name:
        return user_batch_name
    return datetime.now().strftime("batch_%Y%m%d-%H%M%S")


def _find_one_or_more_metrics(run_dir: str) -> list[str]:
    candidates = sorted(glob.glob(os.path.join(run_dir, "**", "metrics.csv"), recursive=True))
    return candidates


def _find_prediction_files(run_dir: str) -> list[str]:
    prefixed = glob.glob(os.path.join(run_dir, "**", "predictions_*.csv"), recursive=True)
    single = glob.glob(os.path.join(run_dir, "**", "predictions.csv"), recursive=True)
    return sorted(set(prefixed + single))


def _has_seed_arg(args: list[str]) -> bool:
    for arg in args:
        if arg == "--seed" or arg.startswith("--seed="):
            return True
    return False


def _mode_from_prediction_path(path: str) -> str:
    name = os.path.basename(path)
    # Ex: predictions_baseline.csv, predictions_few-shot.csv
    if name.startswith("predictions_") and name.endswith(".csv"):
        return name[len("predictions_") : -len(".csv")]
    if name == "predictions.csv":
        return "single"
    return "unknown"


def _stderr_contains_oom(stderr_path: str) -> bool:
    if not os.path.exists(stderr_path):
        return False
    try:
        with open(stderr_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read().lower()
    except OSError:
        return False
    return any(token in content for token in OOM_ERROR_TOKENS)


def _build_oom_retry_args(run_args: list[str], args: argparse.Namespace) -> list[str]:
    retry_args = list(run_args)
    retry_args.extend(
        [
            "--use_quantization",
            "--use_lora",
            "--no-paper_parity",
            "--max_len",
            str(args.oom_retry_max_len),
            "--prompt_budget_tokens",
            str(args.oom_retry_prompt_budget_tokens),
        ]
    )
    return retry_args


def _aggregate_metrics(all_metrics: pd.DataFrame) -> pd.DataFrame:
    # Keep TI and few-shot variants separated; fall back gracefully when columns are absent.
    group_cols = [
        c
        for c in ["ti_mode", "few_shot_mode", "mode", "model", "dataset", "num_shots", "prompt_style"]
        if c in all_metrics.columns
    ]
    if not group_cols:
        raise ValueError(
            "No grouping columns found in metrics; expected at least one grouping key "
            "(e.g., ti_mode/few_shot_mode/model/dataset)."
        )

    metric_cols = [c for c in NUMERIC_METRIC_COLUMNS if c in all_metrics.columns]
    if not metric_cols:
        raise ValueError("No numeric metric columns found to aggregate.")

    grouped = all_metrics.groupby(group_cols, dropna=False)[metric_cols].agg(["mean", "std"]).reset_index()
    grouped.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in grouped.columns.to_flat_index()
    ]
    return grouped


def run_batch(args: argparse.Namespace) -> None:
    repo_root = os.path.abspath(args.repo_root)
    script_path = os.path.abspath(os.path.join(repo_root, args.script_path))

    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Could not find experiment script: {script_path}")

    batch_name = _build_batch_name(args.batch_name)
    batch_root = os.path.abspath(os.path.join(repo_root, args.batch_output_dir, batch_name))
    _ensure_dir(batch_root)

    print(f"Batch root: {batch_root}")
    max_attempts = args.max_attempts if args.max_attempts is not None else args.num_runs
    if max_attempts < args.num_runs:
        raise ValueError("--max_attempts must be >= --num_runs")

    cuda_devices = None if str(args.cuda_visible_devices).lower() == "all" else args.cuda_visible_devices

    print(f"Target successful runs: {args.num_runs}")
    print(f"Max attempts: {max_attempts}")
    print(f"Running experiments from: {script_path}")
    if cuda_devices is not None:
        print(f"Using CUDA_VISIBLE_DEVICES={cuda_devices}")

    run_records = []
    all_metrics_parts = []
    all_predictions_parts = []

    passthrough_args = args.experiment_args or []
    if passthrough_args and passthrough_args[0] == "--":
        passthrough_args = passthrough_args[1:]
    passthrough_has_seed = _has_seed_arg(passthrough_args)
    manifest_path = os.path.join(batch_root, "run_manifest.json")

    def _write_manifest() -> None:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "batch_name": batch_name,
                    "batch_root": batch_root,
                    "script_path": script_path,
                    "python_exec": args.python_exec,
                    "num_runs_requested": args.num_runs,
                    "stop_on_error": args.stop_on_error,
                    "experiment_args": passthrough_args,
                    "runs": run_records,
                },
                f,
                indent=2,
            )

    # Write an initial manifest so progress is inspectable while batch is running.
    _write_manifest()

    if passthrough_has_seed:
        print("Detected explicit --seed in experiment args; batch seed strategy will be ignored.")

    success_count = 0
    attempt_idx = 0
    while success_count < args.num_runs and attempt_idx < max_attempts:
        attempt_idx += 1
        run_idx = attempt_idx
        run_name = f"run_{run_idx:03d}"
        run_dir = os.path.join(batch_root, run_name)
        _ensure_dir(run_dir)

        cmd = [
            args.python_exec,
            script_path,
            "--output_dir",
            run_dir,
        ]

        run_seed = None
        if passthrough_has_seed:
            run_args = list(passthrough_args)
        else:
            run_seed = args.base_seed
            if args.seed_strategy == "increment":
                run_seed = args.base_seed + (attempt_idx - 1)
            run_args = [*passthrough_args, "--seed", str(run_seed)]

        cmd.extend(run_args)

        print("\n" + "=" * 80)
        print(f"[{run_name}] Starting")
        print("Command:", " ".join(cmd))

        run_stdout = os.path.join(run_dir, "run_stdout.log")
        run_stderr = os.path.join(run_dir, "run_stderr.log")

        started_at = datetime.now().isoformat(timespec="seconds")
        run_env = os.environ.copy()
        if cuda_devices is not None:
            run_env["CUDA_VISIBLE_DEVICES"] = cuda_devices
        with open(run_stdout, "w", encoding="utf-8") as out_f, open(run_stderr, "w", encoding="utf-8") as err_f:
            proc = subprocess.run(cmd, cwd=repo_root, stdout=out_f, stderr=err_f, env=run_env)
        finished_at = datetime.now().isoformat(timespec="seconds")

        record = {
            "run_index": run_idx,
            "run_name": run_name,
            "attempt_index": attempt_idx,
            "run_dir": run_dir,
            "command": cmd,
            "started_at": started_at,
            "finished_at": finished_at,
            "return_code": proc.returncode,
            "stdout_path": run_stdout,
            "stderr_path": run_stderr,
            "cuda_visible_devices": cuda_devices,
            "seed_used": run_seed,
        }

        if proc.returncode != 0:
            print(f"[{run_name}] FAILED with return code {proc.returncode}")
            print(f"[{run_name}] See stderr: {run_stderr}")

            if args.enable_oom_retry_fallback and _stderr_contains_oom(run_stderr):
                print(f"[{run_name}] CUDA OOM detected. Retrying once with memory-efficient fallback args.")

                retry_run_args = _build_oom_retry_args(run_args, args)
                retry_cmd = [
                    args.python_exec,
                    script_path,
                    "--output_dir",
                    run_dir,
                ]
                retry_cmd.extend(retry_run_args)

                retry_stdout = os.path.join(run_dir, "run_stdout_oom_retry.log")
                retry_stderr = os.path.join(run_dir, "run_stderr_oom_retry.log")
                retry_started_at = datetime.now().isoformat(timespec="seconds")
                with open(retry_stdout, "w", encoding="utf-8") as out_f, open(retry_stderr, "w", encoding="utf-8") as err_f:
                    retry_proc = subprocess.run(retry_cmd, cwd=repo_root, stdout=out_f, stderr=err_f, env=run_env)
                retry_finished_at = datetime.now().isoformat(timespec="seconds")

                record["oom_retry_applied"] = True
                record["oom_retry_started_at"] = retry_started_at
                record["oom_retry_finished_at"] = retry_finished_at
                record["oom_retry_command"] = retry_cmd
                record["oom_retry_return_code"] = retry_proc.returncode
                record["oom_retry_stdout_path"] = retry_stdout
                record["oom_retry_stderr_path"] = retry_stderr

                if retry_proc.returncode == 0:
                    print(f"[{run_name}] OOM fallback retry succeeded.")
                    record["primary_return_code"] = record["return_code"]
                    record["primary_stdout_path"] = record["stdout_path"]
                    record["primary_stderr_path"] = record["stderr_path"]
                    record["return_code"] = retry_proc.returncode
                    record["command"] = retry_cmd
                    record["stdout_path"] = retry_stdout
                    record["stderr_path"] = retry_stderr
                    proc = retry_proc
                else:
                    print(f"[{run_name}] OOM fallback retry also failed (return code {retry_proc.returncode}).")

            if proc.returncode != 0:
                run_records.append(record)
                _write_manifest()
                if args.stop_on_error:
                    break
                continue

            # OOM fallback succeeded, so continue through normal success handling
            # (metrics discovery, aggregation, and success_count increment).

        metric_paths = _find_one_or_more_metrics(run_dir)
        if not metric_paths:
            print(f"[{run_name}] Completed but no metrics.csv found under {run_dir}")
            run_records.append(record)
            _write_manifest()
            if args.stop_on_error:
                break
            continue

        if len(metric_paths) > 1:
            print(f"[{run_name}] Found multiple metrics files; using all of them.")

        for metric_path in metric_paths:
            metrics_df = pd.read_csv(metric_path)
            metrics_df["run_index"] = run_idx
            metrics_df["run_name"] = run_name
            metrics_df["run_dir"] = run_dir
            metrics_df["metrics_path"] = metric_path
            all_metrics_parts.append(metrics_df)

        pred_paths = _find_prediction_files(run_dir)
        for pred_path in pred_paths:
            pred_df = pd.read_csv(pred_path)
            pred_df["run_index"] = run_idx
            pred_df["run_name"] = run_name
            pred_df["run_dir"] = run_dir
            pred_df["prediction_mode"] = _mode_from_prediction_path(pred_path)
            pred_df["prediction_path"] = pred_path
            all_predictions_parts.append(pred_df)

        print(f"[{run_name}] Completed successfully. Metrics files: {len(metric_paths)} | Prediction files: {len(pred_paths)}")
        success_count += 1
        record["success_index"] = success_count
        run_records.append(record)
        _write_manifest()

    if success_count < args.num_runs:
        print(
            f"Stopped after {attempt_idx} attempts with {success_count} successful runs "
            f"(target was {args.num_runs})."
        )

    batch_incomplete = success_count < args.num_runs

    # Save run manifest regardless of success/failure.
    _write_manifest()

    if not all_metrics_parts:
        print("No metrics were collected from successful runs. See run_manifest.json for details.")
        print(f"Manifest: {manifest_path}")
        if batch_incomplete:
            raise RuntimeError(
                "Batch did not reach the requested number of successful runs; "
                "check run_manifest.json and per-run stderr logs."
            )
        return

    all_metrics_df = pd.concat(all_metrics_parts, ignore_index=True)
    all_metrics_path = os.path.join(batch_root, "all_runs_metrics.csv")
    all_metrics_df.to_csv(all_metrics_path, index=False)

    summary_df = _aggregate_metrics(all_metrics_df)
    summary_path = os.path.join(batch_root, "aggregate_metrics_mean_std.csv")
    summary_df.to_csv(summary_path, index=False)

    if all_predictions_parts:
        all_predictions_df = pd.concat(all_predictions_parts, ignore_index=True)
        all_predictions_path = os.path.join(batch_root, "all_runs_predictions.csv")
        all_predictions_df.to_csv(all_predictions_path, index=False)
    else:
        all_predictions_path = None

    print("\n" + "=" * 80)
    print("Batch complete.")
    print(f"Manifest: {manifest_path}")
    print(f"All metrics: {all_metrics_path}")
    print(f"Aggregate mean/std: {summary_path}")
    if all_predictions_path:
        print(f"All predictions: {all_predictions_path}")

    if batch_incomplete:
        raise RuntimeError(
            "Batch produced partial results but did not reach the requested "
            "number of successful runs."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run thinking_intervention.py multiple times and aggregate metrics across runs."
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=3,
        help="Target number of successful completed runs.",
    )
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=None,
        help="Maximum total attempts (including failures). Defaults to --num_runs.",
    )
    parser.add_argument(
        "--repo_root",
        type=str,
        default=".",
        help="Repository root used as working directory for each run.",
    )
    parser.add_argument(
        "--script_path",
        type=str,
        default="thinking_intervention.py",
        help="Path to experiment script, relative to repo_root unless absolute.",
    )
    parser.add_argument(
        "--batch_output_dir",
        type=str,
        default="Results_TI/batch_runs",
        help="Directory (under repo_root) where batch outputs are stored.",
    )
    parser.add_argument(
        "--batch_name",
        type=str,
        default=None,
        help="Optional batch folder name. If omitted, a timestamped name is used.",
    )
    parser.add_argument(
        "--python_exec",
        type=str,
        default=sys.executable,
        help="Python executable used to launch each experiment run.",
    )
    parser.add_argument(
        "--cuda_visible_devices",
        type=str,
        default="0",
        help="Value for CUDA_VISIBLE_DEVICES for each run (use 'all' to disable pinning).",
    )
    parser.add_argument(
        "--stop_on_error",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stop batch immediately if any run fails.",
    )
    parser.add_argument(
        "--seed_strategy",
        type=str,
        choices=["fixed", "increment"],
        default="increment",
        help="Seed policy when --seed is not explicitly passed through experiment args.",
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=42,
        help="Base seed used by batch seed strategy.",
    )
    parser.add_argument(
        "--enable_oom_retry_fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If a run fails with CUDA OOM, retry once with memory-efficient overrides "
            "(--use_quantization --use_lora --no-paper_parity plus shorter lengths)."
        ),
    )
    parser.add_argument(
        "--oom_retry_max_len",
        type=int,
        default=192,
        help="max_len used for OOM fallback retry.",
    )
    parser.add_argument(
        "--oom_retry_prompt_budget_tokens",
        type=int,
        default=192,
        help="prompt_budget_tokens used for OOM fallback retry.",
    )
    parser.add_argument(
        "experiment_args",
        nargs=argparse.REMAINDER,
        help=(
            "Any arguments after '--' are passed through to thinking_intervention.py. "
            "Example: -- --ti_mode structure_focus --few_shot_mode infer"
        ),
    )

    cli_args = parser.parse_args()

    if cli_args.num_runs < 1:
        raise ValueError("--num_runs must be >= 1")

    run_batch(cli_args)
