"""
Evaluate Gemma/Qwen with token-generation-level thought intervention.

This script does NOT modify chain-of-thought behavior via prompt instructions.
Instead, it intervenes at decoding time through Ollama generation options
(mainly stop-token controls and decoding constraints).

Usage examples
--------------
# Baseline vs token-level no-think intervention for both families
python LLMS/eval_ollama_token_intervention.py \
  --models gemma3:12b qwen3:8b \
  --token_interventions none block_think_tokens

# Add stronger decoding constraints during intervention
python LLMS/eval_ollama_token_intervention.py \
  --models qwen3:8b \
  --token_interventions none block_think_tokens_strict \
  --max_samples 200
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from difflib import get_close_matches
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

TEXT_COL = "source_article"
LABEL_COL = "logical_fallacies"

SUPPORTED_FAMILIES = ("gemma", "qwen")


@dataclass
class Prediction:
    pred_label: str
    raw_response: str
    parsed_ok: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run zero-shot logical-fallacy classification with token-level intervention for Gemma/Qwen.",
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
        help="Ollama model tags from Gemma/Qwen families, e.g. gemma3:12b qwen3:8b",
    )
    parser.add_argument(
        "--token_interventions",
        nargs="+",
        default=["none", "block_think_tokens"],
        choices=["none", "block_think_tokens", "block_think_tokens_strict"],
        help="Decoding-time intervention policies to run. Include 'none' for baseline deltas.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="Results",
        help="Root directory where metrics and predictions are written.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Limit rows per split for quick experiments. 0 means use all rows.",
    )
    parser.add_argument(
        "--ollama_host",
        type=str,
        default="",
        help="Optional Ollama host, e.g. http://127.0.0.1:11434",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Baseline sampling temperature.",
    )

    return parser.parse_args()


def normalized(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def build_prompt(text: str, labels: list[str]) -> str:
    labels_text = "\n".join(f"- {lbl}" for lbl in labels)
    return (
        "Classify the logical fallacy in the text.\n"
        "Choose exactly one label from this list:\n"
        f"{labels_text}\n\n"
        "Return ONLY valid JSON with this schema: "
        '{"label": "<one label from list>", "reason": "<short reason>"}.\n\n'
        f"Text:\n{text}"
    )


def extract_json_obj(text: str) -> dict[str, Any] | None:
    text = text.strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None

    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def resolve_label(candidate: str, valid_labels: list[str], norm_map: dict[str, str]) -> str | None:
    if not candidate:
        return None

    cand_norm = normalized(candidate)
    if cand_norm in norm_map:
        return norm_map[cand_norm]

    for lbl_norm, lbl in norm_map.items():
        if cand_norm == lbl_norm or cand_norm in lbl_norm or lbl_norm in cand_norm:
            return lbl

    close = get_close_matches(cand_norm, list(norm_map.keys()), n=1, cutoff=0.7)
    if close:
        return norm_map[close[0]]

    for lbl in valid_labels:
        if lbl.lower() in candidate.lower():
            return lbl

    return None


def infer_family(model: str) -> str:
    m = model.lower()
    if "gemma" in m:
        return "gemma"
    if "qwen" in m:
        return "qwen"
    return "other"


def family_stop_tokens(model: str) -> list[str]:
    family = infer_family(model)

    common = ["<think>", "</think>"]
    if family == "qwen":
        return common + ["<|im_start|>think", "<|im_end|>"]

    if family == "gemma":
        return common

    return common


def build_generation_options(model: str, temperature: float, token_intervention: str) -> dict[str, Any]:
    options: dict[str, Any] = {
        "temperature": temperature,
    }

    if token_intervention == "none":
        return options

    stops = family_stop_tokens(model)
    options["stop"] = stops

    if token_intervention == "block_think_tokens_strict":
        # Strict decoding settings reduce drift while still using token-level controls.
        options["temperature"] = 0.0
        options["top_p"] = 0.9
        options["top_k"] = 20

    return options


def predict_one(
    client: Any,
    model: str,
    text: str,
    labels: list[str],
    norm_map: dict[str, str],
    temperature: float,
    token_intervention: str,
) -> Prediction:
    system = "You are a precise classifier. Follow instructions exactly and return only JSON."
    prompt = build_prompt(text, labels)

    options = build_generation_options(
        model=model,
        temperature=temperature,
        token_intervention=token_intervention,
    )

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "options": options,
    }

    try:
        response = client.chat(format="json", **kwargs)
    except TypeError:
        response = client.chat(**kwargs)

    raw = response["message"]["content"].strip()

    obj = extract_json_obj(raw)
    if obj is None:
        return Prediction(pred_label="__UNPARSEABLE__", raw_response=raw, parsed_ok=False)

    candidate = ""
    for key in ("label", "prediction", "class", "logical_fallacy"):
        if key in obj and isinstance(obj[key], str):
            candidate = obj[key]
            break

    pred = resolve_label(candidate, labels, norm_map)
    if pred is None:
        return Prediction(pred_label="__UNPARSEABLE__", raw_response=raw, parsed_ok=False)

    return Prediction(pred_label=pred, raw_response=raw, parsed_ok=True)


def score_split(true_labels: list[str], pred_labels: list[str]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(true_labels, pred_labels)),
        "precision_macro": float(precision_score(true_labels, pred_labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(true_labels, pred_labels, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(true_labels, pred_labels, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(true_labels, pred_labels, average="weighted", zero_division=0)),
    }


def build_experiment_tag(data_tag: str, models: list[str], max_samples: int) -> str:
    model_tag = "+".join(m.replace("/", "_").replace(":", "_") for m in models)
    sample_tag = f"n{max_samples}" if max_samples > 0 else "nall"
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"data-{data_tag}__ollama-token-{model_tag}__{sample_tag}__{ts}"


def load_split(path: str, max_samples: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    if TEXT_COL not in df.columns or LABEL_COL not in df.columns:
        raise ValueError(
            f"Missing required columns in {path}. Expected '{TEXT_COL}' and '{LABEL_COL}'."
        )

    df = df[[TEXT_COL, LABEL_COL]].dropna()
    if max_samples > 0:
        df = df.head(max_samples)

    return df.reset_index(drop=True)


def run_model_on_split(
    client: Any,
    model: str,
    split_name: str,
    df: pd.DataFrame,
    labels: list[str],
    temperature: float,
    token_intervention: str,
) -> tuple[dict[str, float], pd.DataFrame, float]:
    norm_map = {normalized(lbl): lbl for lbl in labels}

    records: list[dict[str, Any]] = []
    start = time.time()

    for row_num, (_, row) in enumerate(
        tqdm(
            df.iterrows(),
            total=len(df),
            desc=f"[{model}][{token_intervention}][{split_name}]",
            ncols=100,
        )
    ):
        pred = predict_one(
            client=client,
            model=model,
            text=str(row[TEXT_COL]),
            labels=labels,
            norm_map=norm_map,
            temperature=temperature,
            token_intervention=token_intervention,
        )

        true_label = str(row[LABEL_COL])
        records.append(
            {
                "idx": row_num,
                "text": str(row[TEXT_COL]),
                "true_label": true_label,
                "pred_label": pred.pred_label,
                "correct": int(pred.pred_label == true_label),
                "parsed_ok": int(pred.parsed_ok),
                "raw_response": pred.raw_response,
            }
        )

    elapsed = time.time() - start
    pred_df = pd.DataFrame(records)
    metrics = score_split(pred_df["true_label"].tolist(), pred_df["pred_label"].tolist())
    metrics["parsed_rate"] = float(pred_df["parsed_ok"].mean()) if len(pred_df) else 0.0
    metrics["num_samples"] = float(len(pred_df))
    metrics["seconds"] = round(elapsed, 3)
    metrics["samples_per_sec"] = round((len(pred_df) / elapsed), 3) if elapsed > 0 else 0.0

    print(
        f"[{model}][{token_intervention}][{split_name}] "
        f"acc={metrics['accuracy']:.4f} f1_macro={metrics['f1_macro']:.4f} "
        f"parsed={metrics['parsed_rate']:.3f} n={int(metrics['num_samples'])} time={metrics['seconds']:.1f}s"
    )

    return metrics, pred_df, elapsed


def compute_delta_vs_none(summary_df: pd.DataFrame) -> pd.DataFrame:
    baseline = summary_df[summary_df["token_intervention"] == "none"].copy()
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

    baseline = baseline[key_cols + metric_cols].rename(
        columns={c: f"baseline_{c}" for c in metric_cols}
    )

    merged = summary_df.merge(baseline, on=key_cols, how="left")
    for c in metric_cols:
        merged[f"delta_{c}"] = merged[c] - merged[f"baseline_{c}"]

    return merged


def main() -> None:
    args = parse_args()

    for model in args.models:
        family = infer_family(model)
        if family not in SUPPORTED_FAMILIES:
            raise ValueError(
                f"Model '{model}' is not from Gemma/Qwen families. "
                "Use model tags containing 'gemma' or 'qwen'."
            )

    try:
        from ollama import Client  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError("Missing Python package 'ollama'. Install with: pip install ollama") from exc

    train_path = os.path.join(args.data_dir, args.train_file)
    dev_path = os.path.join(args.data_dir, args.dev_file)
    test_path = os.path.join(args.data_dir, args.test_file)

    for p in (train_path, dev_path, test_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Data file not found: {p}")

    train_df = load_split(train_path, max_samples=0)
    dev_df = load_split(dev_path, max_samples=args.max_samples)
    test_df = load_split(test_path, max_samples=args.max_samples)

    labels = sorted(train_df[LABEL_COL].astype(str).unique().tolist())
    data_tag = os.path.splitext(args.train_file)[0].replace("_train", "")

    exp_tag = build_experiment_tag(data_tag=data_tag, models=args.models, max_samples=args.max_samples)
    exp_dir = os.path.join(args.output_dir, exp_tag)
    pred_dir = os.path.join(exp_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    if args.ollama_host:
        client = Client(host=args.ollama_host)
    else:
        client = Client()

    summary_rows: list[dict[str, Any]] = []

    interventions = list(dict.fromkeys(args.token_interventions))

    for model in args.models:
        print(f"\n=== Running model: {model} ===")
        for token_intervention in interventions:
            for split_name, split_df in (("dev", dev_df), ("test", test_df)):
                metrics, pred_df, _ = run_model_on_split(
                    client=client,
                    model=model,
                    split_name=split_name,
                    df=split_df,
                    labels=labels,
                    temperature=args.temperature,
                    token_intervention=token_intervention,
                )

                pred_path = os.path.join(
                    pred_dir,
                    f"{model.replace('/', '_').replace(':', '_')}_{token_intervention}_{split_name}_predictions.csv",
                )
                pred_df.to_csv(pred_path, index=False)

                row = {
                    "model": model,
                    "split": split_name,
                    "temperature": args.temperature,
                    "token_intervention": token_intervention,
                    "num_labels": len(labels),
                }
                row.update(metrics)
                summary_rows.append(row)

    metrics_path = os.path.join(exp_dir, "metrics.csv")
    if summary_rows:
        keys: list[str] = []
        for r in summary_rows:
            for k in r.keys():
                if k not in keys:
                    keys.append(k)

        with open(metrics_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(summary_rows)

    summary_df = pd.DataFrame(summary_rows)
    delta_df = compute_delta_vs_none(summary_df)

    delta_path = os.path.join(exp_dir, "delta_vs_none.csv")
    if not delta_df.empty:
        delta_df.to_csv(delta_path, index=False)

    print("\nSaved:")
    print(f"- Metrics: {metrics_path}")
    print(f"- Predictions: {pred_dir}")
    if not delta_df.empty:
        print(f"- Delta vs none: {delta_path}")
    else:
        print("- Delta vs none: skipped (no 'none' in --token_interventions)")


if __name__ == "__main__":
    main()
