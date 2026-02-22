"""
Train BERT-family baseline models on the logical-fallacy classification task.

Usage examples
--------------
# Train all default models on the combined dataset:
    python train_baselines.py

# Train a single model:
    python train_baselines.py --models bert-base-uncased

# Train on the edu subset only:
    python train_baselines.py --data_dir Data/edu_data \
        --train_file logic_edu_train.csv --dev_file logic_edu_dev.csv --test_file logic_edu_test.csv

# Override training hyper-parameters:
    python train_baselines.py --epochs 5 --batch_size 32 --lr 3e-5 --max_length 512
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from BaselineModels.HFModel import HFModel


# ── Default BERT variants to benchmark ──────────────────────────────────────
DEFAULT_MODELS = [
    "bert-base-uncased",
    "distilbert-base-uncased",
    "roberta-base",
    "albert-base-v2",
]


def build_experiment_tag(args: argparse.Namespace) -> str:
    """Build a human-readable tag that encodes the experiment configuration.

    Example output:
        data-all_logic__bert-base-uncased+roberta-base__ep3_bs16_lr2e-05_ml256
    """
    # ── Data tag: derive from train filename (drop _train.csv suffix) ────
    data_tag = os.path.splitext(args.train_file)[0]          # all_logic_train
    for suffix in ["_train", "_test", "_dev"]:
        data_tag = data_tag.replace(suffix, "")              # all_logic
    # If data_dir is a subdirectory (e.g. Data/edu_data), prepend it
    data_dir_base = os.path.basename(os.path.normpath(args.data_dir))
    if data_dir_base.lower() not in ("data", "."):
        data_tag = f"{data_dir_base}__{data_tag}"

    # ── Model tag: short names joined with '+' ──────────────────────────
    short_names = [m.replace("/", "_") for m in args.models]
    model_tag = "+".join(short_names)

    # ── Hyper-param tag ─────────────────────────────────────────────────
    hp_tag = f"ep{args.epochs}_bs{args.batch_size}_lr{args.lr}_ml{args.max_length}"

    return f"data-{data_tag}__{model_tag}__{hp_tag}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune BERT-family baselines on logical-fallacy detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--data_dir",
        type=str,
        default="Data",
        help="Root directory containing the CSV splits.",
    )
    parser.add_argument("--train_file", type=str, default="all_logic_train.csv")
    parser.add_argument("--dev_file", type=str, default="all_logic_dev.csv")
    parser.add_argument("--test_file", type=str, default="all_logic_test.csv")

    # ── Models ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="HuggingFace model identifiers to train.",
    )

    # ── Training hyper-parameters ───────────────────────────────────────────
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_length", type=int, default=256, help="Max tokenizer length.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps. Increase to simulate larger batches with less memory.",
    )

    # ── Output ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--output_dir",
        type=str,
        default="Results",
        help="Directory for saved models, metrics CSVs, and predictions.",
    )
    parser.add_argument(
        "--save_models",
        action="store_true",
        default=False,
        help="Persist fine-tuned model weights to disk.",
    )

    return parser.parse_args()


def evaluate_predictions(
    true_ids: list[int],
    pred_ids: list[int],
    id2label: dict[int, str],
) -> dict:
    """Return a dict of aggregate metrics for a single model."""
    labels_present = sorted(set(true_ids) | set(pred_ids))
    target_names = [id2label.get(i, str(i)) for i in labels_present]

    acc = accuracy_score(true_ids, pred_ids)
    prec_macro = precision_score(true_ids, pred_ids, average="macro", zero_division=0)
    rec_macro = recall_score(true_ids, pred_ids, average="macro", zero_division=0)
    f1_macro = f1_score(true_ids, pred_ids, average="macro", zero_division=0)
    f1_weighted = f1_score(true_ids, pred_ids, average="weighted", zero_division=0)

    report_str = classification_report(
        true_ids, pred_ids, labels=labels_present,
        target_names=target_names, zero_division=0,
    )

    return {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "classification_report": report_str,
    }


def run_one_model(
    model_name: str,
    args: argparse.Namespace,
    train_path: str,
    dev_path: str,
    test_path: str,
    experiment_dir: str,
) -> dict:
    """Train a single model, evaluate on dev & test, and return metrics dict."""

    print(f"\n{'='*70}")
    print(f"  Model: {model_name}")
    print(f"{'='*70}")

    model_safe_name = model_name.replace("/", "_")
    model_output_dir = os.path.join(experiment_dir, "checkpoints", model_safe_name)

    hf = HFModel(
        model_name=model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        output_dir=model_output_dir,
        seed=args.seed,
    )
    hf.gradient_accumulation_steps = args.grad_accum_steps

    # ── Data ────────────────────────────────────────────────────────────────
    # Train: used to update model weights.
    # Dev:   used for checkpoint selection during training (load_best_model_at_end).
    #        Dev metrics are logged for monitoring but are optimistic because the
    #        best checkpoint was *chosen* to maximise dev accuracy.
    # Test:  held-out final evaluation — never influences training or model selection.
    print("  Loading data …")
    hf.load_train_dev_test_data(train_path, dev_path, test_path)
    num_labels = len(hf.label2id)
    print(f"  Labels ({num_labels}): {list(hf.label2id.keys())}")

    # ── Train ───────────────────────────────────────────────────────────────
    print("  Training …")
    start = time.time()
    hf.train()
    train_seconds = time.time() - start
    print(f"  Training completed in {train_seconds:.1f}s")

    # ── Predict on dev & test ───────────────────────────────────────────────
    results: dict = {
        "model": model_name,
        "num_labels": num_labels,
        "train_samples": len(hf.train_texts),
        "dev_samples": len(hf.dev_texts),
        "test_samples": len(hf.test_texts),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "max_length": args.max_length,
        "train_time_s": round(train_seconds, 2),
    }

    for split_name, texts, true_labels in [
        ("dev", hf.dev_texts, hf.dev_labels),
        ("test", hf.test_texts, hf.test_labels),
    ]:
        if not texts or true_labels is None:
            continue

        preds = hf.predict(texts)
        if isinstance(preds, dict):  # single-sample edge case
            preds = [preds]
        pred_ids = [p["label_id"] for p in preds]

        metrics = evaluate_predictions(true_labels, pred_ids, hf.id2label)

        tag = "(monitoring)" if split_name == "dev" else "(final eval)"
        print(f"\n  ── {split_name.upper()} results {tag} ──")
        print(f"  Accuracy : {metrics['accuracy']:.4f}")
        print(f"  F1 macro : {metrics['f1_macro']:.4f}")
        print(f"  F1 wt.   : {metrics['f1_weighted']:.4f}")
        print(metrics["classification_report"])

        # Flatten into the results dict with a split prefix
        for key in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]:
            results[f"{split_name}_{key}"] = round(metrics[key], 6)

        # Save per-sample predictions to CSV
        preds_df = pd.DataFrame(
            {
                "text": texts,
                "true_label": [hf.id2label[tid] for tid in true_labels],
                "pred_label": [p["label"] for p in preds],
                "pred_score": [round(p["score"], 6) for p in preds],
                "correct": [int(t == p["label_id"]) for t, p in zip(true_labels, preds)],
            }
        )
        preds_csv = os.path.join(
            experiment_dir, "predictions", f"{model_safe_name}_{split_name}_predictions.csv"
        )
        os.makedirs(os.path.dirname(preds_csv), exist_ok=True)
        preds_df.to_csv(preds_csv, index=False)
        print(f"  Predictions saved → {preds_csv}")

    # ── Save model (optional) ──────────────────────────────────────────────
    if args.save_models:
        save_dir = os.path.join(experiment_dir, "saved_models", model_safe_name)
        hf.save_model(save_dir)
        print(f"  Model saved → {save_dir}")

    # ── Free memory before the next model ────────────────────────────────
    hf.cleanup()

    return results


def main():
    args = parse_args()

    train_path = os.path.join(args.data_dir, args.train_file)
    dev_path = os.path.join(args.data_dir, args.dev_file)
    test_path = os.path.join(args.data_dir, args.test_file)

    for path in [train_path, dev_path, test_path]:
        if not os.path.isfile(path):
            print(f"ERROR: data file not found: {path}", file=sys.stderr)
            sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Build experiment tag and create a dedicated subdirectory ──────────
    tag = build_experiment_tag(args)
    experiment_dir = os.path.join(args.output_dir, tag)
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"Experiment directory: {experiment_dir}")

    all_results: list[dict] = []

    for model_name in args.models:
        try:
            result = run_one_model(model_name, args, train_path, dev_path, test_path, experiment_dir)
            all_results.append(result)
        except Exception as exc:
            print(f"\n  !! {model_name} failed: {exc}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            all_results.append({"model": model_name, "error": str(exc)})

    # ── Write aggregate metrics CSV ─────────────────────────────────────────
    metrics_csv = os.path.join(experiment_dir, "metrics.csv")
    if all_results:
        # Collect all keys across runs (some models may have failed)
        all_keys: list[str] = []
        for r in all_results:
            for k in r:
                if k not in all_keys:
                    all_keys.append(k)

        with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(all_results)

        print(f"\n{'='*70}")
        print(f"  Aggregate metrics saved → {metrics_csv}")
        print(f"{'='*70}")

    # ── Quick summary table ─────────────────────────────────────────────────
    summary_cols = ["model", "dev_accuracy", "dev_f1_macro", "test_accuracy", "test_f1_macro", "train_time_s"]
    summary_data = [
        {k: r.get(k, "") for k in summary_cols}
        for r in all_results
        if "error" not in r
    ]
    if summary_data:
        print("\n  Summary  (dev = monitoring only; test = final evaluation)")
        print("  " + "-" * 90)
        header = f"  {'Model':<30} {'Dev Acc':>8} {'Dev F1':>8} {'Test Acc':>9} {'Test F1':>8} {'Time(s)':>8}"
        print(header)
        print("  " + "-" * 90)
        for row in summary_data:
            print(
                f"  {row['model']:<30} "
                f"{row.get('dev_accuracy', ''):>8} "
                f"{row.get('dev_f1_macro', ''):>8} "
                f"{row.get('test_accuracy', ''):>9} "
                f"{row.get('test_f1_macro', ''):>8} "
                f"{row.get('train_time_s', ''):>8}"
            )
        print()


if __name__ == "__main__":
    main()
