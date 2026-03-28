"""
Evaluate one or more Ollama LLMs as zero-shot baselines on logical fallacy data.

Usage examples
--------------
# Evaluate one model on all_logic dev + test
python LLMS/eval_ollama_baseline.py --models llama3.2:3b

# Evaluate two models with a small sample for a quick smoke test
python LLMS/eval_ollama_baseline.py --models llama3.2:3b mistral:7b --max_samples 100

# Point at a custom Ollama server
python LLMS/eval_ollama_baseline.py --models llama3.2:3b --ollama_host http://127.0.0.1:11434
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


TEXT_COL = "source_article"
LABEL_COL = "logical_fallacies"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run zero-shot logical-fallacy classification with Ollama models.",
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
        help="Ollama model tags, e.g. llama3.2:3b mistral:7b",
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
        help="Sampling temperature. Use 0 for deterministic baselines.",
    )

    return parser.parse_args()


def normalized(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


@dataclass
class Prediction:
    pred_label: str
    raw_response: str
    parsed_ok: bool


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


def predict_one(
    client: Any,
    model: str,
    text: str,
    labels: list[str],
    norm_map: dict[str, str],
    temperature: float,
) -> Prediction:
    system = (
        "You are a precise classifier. "
        "Follow instructions exactly and return only JSON."
    )
    prompt = build_prompt(text, labels)

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "options": {"temperature": temperature},
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
    return f"data-{data_tag}__ollama-{model_tag}__{sample_tag}__{ts}"


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
) -> tuple[dict[str, float], pd.DataFrame, float]:
    norm_map = {normalized(lbl): lbl for lbl in labels}

    records: list[dict[str, Any]] = []
    start = time.time()

    for row_num, (_, row) in enumerate(df.iterrows()):
        pred = predict_one(
            client=client,
            model=model,
            text=str(row[TEXT_COL]),
            labels=labels,
            norm_map=norm_map,
            temperature=temperature,
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
        f"[{model}][{split_name}] "
        f"acc={metrics['accuracy']:.4f} f1_macro={metrics['f1_macro']:.4f} "
        f"parsed={metrics['parsed_rate']:.3f} n={int(metrics['num_samples'])} "
        f"time={metrics['seconds']:.1f}s"
    )

    return metrics, pred_df, elapsed


def main() -> None:
    args = parse_args()

    try:
        from ollama import Client  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "Missing Python package 'ollama'. Install with: pip install ollama"
        ) from exc

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

    for model in args.models:
        print(f"\n=== Running model: {model} ===")

        for split_name, split_df in (("dev", dev_df), ("test", test_df)):
            metrics, pred_df, _ = run_model_on_split(
                client=client,
                model=model,
                split_name=split_name,
                df=split_df,
                labels=labels,
                temperature=args.temperature,
            )

            pred_path = os.path.join(
                pred_dir,
                f"{model.replace('/', '_').replace(':', '_')}_{split_name}_predictions.csv",
            )
            pred_df.to_csv(pred_path, index=False)

            row = {
                "model": model,
                "split": split_name,
                "temperature": args.temperature,
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

    print("\nSaved:")
    print(f"- Metrics: {metrics_path}")
    print(f"- Predictions: {pred_dir}")


if __name__ == "__main__":
    main()
