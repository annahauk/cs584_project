"""
thinking_intervention.py

Adds Thinking Intervention (Wu et al. 2025, arXiv:2503.24370) on top of
the few-shot replication baseline from paper_replication.py.

Architecture overview
---------------------
paper_replication.py has two few-shot paths:

  (A) few_shot_mode="infer"  — log-prob scoring via AutoModelForCausalLM.
        The prompt ends with an assistant generation header. Label likelihood
        is computed as mean log P(label_token | prompt_prefix). TI works by
        extending that prefix with a first-person reasoning statement BEFORE
        label scoring, i.e. the model scores each label conditioned on the
        intervention reasoning. This is the TIbegin strategy from Wu et al.
        §2.3 — intervening at the START of the reasoning process — which the
        paper shows consistently outperforms mid and end-of-reasoning variants.

  (B) few_shot_mode="train"  — builds prompted texts and fine-tunes a
        AutoModelForSequenceClassification head. TI here appends the
        intervention as a reasoning prefix to every prompted training sample,
        so the classifier learns to use the reasoning context as a feature.
        This is less principled than path A but is included for completeness.

Model-family handling
---------------------
- Qwen3 / DeepSeek R1 reasoning models use <think>...</think> tags.
  TI appends "<think>{prefix}" to the assistant turn.
- Standard instruct models (Qwen2.5-Instruct, Gemma-2-it, Phi-3) have no
  reasoning tags. TI appends the prefix as plain text to the assistant turn.
  The label is still scored conditioned on this reasoning context.
- Gemma models reject an explicit system role in apply_chat_template.
  The same fallback used in paper_replication._is_system_role_unsupported_error
  is reused here.

Usage
-----
# Exact baseline (should reproduce paper_replication few-shot infer results)
python thinking_intervention.py --ti_mode none --few_shot_mode infer

# Main TI contribution — structure_focus targets equivocation / intentional
python thinking_intervention.py --ti_mode structure_focus --few_shot_mode infer

# Ablation across all modes
for mode in none label_first structure_focus counterfactual suppress_cot; do
    python thinking_intervention.py --ti_mode $mode --few_shot_mode infer
done
"""
from __future__ import annotations

import argparse
import os
import time
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    set_seed,
)

# ── Import everything reusable from the replication baseline ─────────────────
# We import private helpers too (_-prefixed) because they are stable
# implementation details we need to reuse exactly, not re-implement.
from paper_replication import (
    # data
    load_dataset,
    preprocess,
    encode_labels,
    # prompt building
    build_few_shot_prompt,
    build_few_shot_prompt_budgeted,
    build_prompted_texts,
    _truncate_text_to_tokens,
    _select_topk_indices,
    # model helpers
    _ensure_tokenizer_pad_token,
    _is_system_role_unsupported_error,
    _infer_lora_target_modules,          # imported for train-mode TI
    # metrics / output
    compute_metrics,
    plot_confusion,
    build_experiment_tag,
    print_gpu_utilization,
    # training path (used for few_shot_mode="train" TI variant)
    FallacyDataset,
    train_model,
    evaluate,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. MODEL FAMILY DETECTION
# ─────────────────────────────────────────────────────────────────────────────

# Models known to support explicit <think>...</think> reasoning tags.
# Extend this set as you add new reasoning models.
_REASONING_MODEL_SUBSTRINGS = (
    "deepseek-r1",
    "r1-",
    "qwq",
    "qwen3",           # Qwen3 series supports thinking mode
)

def _model_supports_think_tags(model_name: str) -> bool:
    """
    Returns True if the model uses explicit <think>...</think> reasoning blocks.
    Standard instruct models (Qwen2.5-Instruct, Gemma-2-it, Phi-3) do not.
    """
    name_lower = model_name.lower()
    return any(s in name_lower for s in _REASONING_MODEL_SUBSTRINGS)


# ─────────────────────────────────────────────────────────────────────────────
# 2. THINKING INTERVENTION SEQUENCES
#
# Paper §3: interventions are first-person narrative injected at the START
# of the reasoning process (TIbegin). Wu et al. show this consistently
# outperforms middle and end-of-reasoning intervention positions (§6.2).
#
# Intervention text is grounded in Teo et al.'s findings:
#   - Models do well on "ad hominem", "ad populum", "false causality"
#   - Models struggle with "equivocation", "intentional", "fallacy of relevance"
#   - Confusion arises from semantic overlap between emotionally charged fallacies
# ─────────────────────────────────────────────────────────────────────────────

TI_MODES = ("none", "label_first", "structure_focus", "counterfactual", "suppress_cot")


def build_thinking_prefix(ti_mode: str, labels: list[str]) -> str:
    """
    Returns a first-person intervention string to inject into the assistant turn.

    For reasoning models this is appended after <think>.
    For standard instruct models it is appended directly after the assistant header.
    In both cases it conditions label-scoring on the intervention reasoning.

    Args:
        ti_mode:  One of TI_MODES.
        labels:   The full list of fallacy label strings for this dataset split.

    Returns:
        Intervention string, or "" for ti_mode="none".
    """
    if ti_mode not in TI_MODES:
        raise ValueError(f"Unknown ti_mode {ti_mode!r}. Must be one of: {TI_MODES}")

    if ti_mode == "none":
        return ""

    label_list = ", ".join(f'"{l}"' for l in labels)

    if ti_mode == "label_first":
        # Anchors the model on the output constraint immediately.
        # Addresses the label hallucination failure mode seen in zero-shot baselines.
        return (
            f"I must select exactly one label from this list: {label_list}. "
            "I will not invent new categories or return an empty label. "
            "Let me identify the argument structure first, then match it to the correct fallacy."
        )

    if ti_mode == "structure_focus":
        # Directly targets Teo et al.'s finding: all models confuse semantically
        # similar fallacies — equivocation vs appeal to emotion vs fallacy of relevance.
        # Forces attention to logical form rather than surface topic or emotional tone.
        return (
            "I need to focus on the LOGICAL STRUCTURE of this argument, not its topic or "
            "emotional tone. "
            "I should ask: what specific reasoning error is being made? "
            "Emotional language does not automatically mean 'appeal to emotion' if the "
            "underlying error is a different structural flaw. "
            "I need to carefully distinguish between fallacies that share surface features. "
            f"The label must be exactly one of: {label_list}."
        )

    if ti_mode == "counterfactual":
        # Paper §6.2 counterfactual check: identify best candidate, then explicitly
        # reject the next most plausible alternative before committing.
        return (
            "I will identify my top candidate fallacy label, then consider the next most "
            "plausible alternative and articulate why I am rejecting it before making "
            "my final decision. "
            f"Available labels: {label_list}."
        )

    if ti_mode == "suppress_cot":
        # Useful for models that over-reason and circle back to wrong labels.
        # Mirrors the overthinking mitigation result in Wu et al. Appendix C.
        return (
            "I will identify the fallacy type directly without extended deliberation. "
            f"I need to pick exactly one label from: {label_list} and commit immediately."
        )

    # Should never reach here given the guard above, but keeps type-checkers happy.
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# 3. PROMPT BUILDER WITH THINKING INTERVENTION
#
# Wraps build_few_shot_prompt_budgeted (from paper_replication) and appends
# the TI prefix as a partial assistant turn. The Gemma system-role fallback
# is handled upstream in build_few_shot_prompt, so the string returned by
# build_few_shot_prompt_budgeted already ends correctly regardless of model.
# We just need to append the thinking prefix after that assistant header.
# ─────────────────────────────────────────────────────────────────────────────

def build_few_shot_prompt_with_ti(
    text: str,
    examples: list[tuple[str, str]],
    labels: list[str],
    tokenizer,
    ti_mode: str,
    model_name: str,
    max_prompt_tokens: int = 1024,
    prompt_style: str = "hybrid",
) -> str:
    """
    Builds a budgeted few-shot prompt, then appends the TI prefix as a
    partial assistant reasoning turn (TIbegin strategy, Wu et al. §2.3).

    The returned string ends with:
        <|im_start|>assistant\n<think>{thinking_prefix}   (reasoning models)
        <|im_start|>assistant\n{thinking_prefix}           (instruct models)

    The model then continues generating FROM the intervention — or in the
    log-prob scoring path, labels are scored conditioned on this context.

    Args:
        text:             Input text to classify.
        examples:         Few-shot (text, label) pairs.
        labels:           Full label list.
        tokenizer:        HuggingFace tokenizer.
        ti_mode:          Intervention mode string.
        model_name:       HuggingFace model name/path (used for tag detection).
        max_prompt_tokens: Token budget for the base prompt.
        prompt_style:     "minimal" or "hybrid".

    Returns:
        Prompt string with TI prefix appended, or plain base prompt for "none".
    """
    base_prompt = build_few_shot_prompt_budgeted(
        text,
        examples,
        labels,
        tokenizer,
        max_prompt_tokens=max_prompt_tokens,
        prompt_style=prompt_style,
    )

    thinking_prefix = build_thinking_prefix(ti_mode, labels)
    if not thinking_prefix:
        return base_prompt

    if _model_supports_think_tags(model_name):
        # Reasoning model: wrap prefix in <think> so the model continues
        # its internal chain-of-thought from the intervention point.
        return base_prompt + f"<think>{thinking_prefix}\n"
    else:
        # Standard instruct model: append as plain reasoning context.
        # Label log-probs are still conditioned on this text, which is the
        # mechanism by which TI works for non-reasoning models.
        return base_prompt + f"{thinking_prefix}\n"


# ─────────────────────────────────────────────────────────────────────────────
# 4. FEW-SHOT INFERENCE WITH TI  (few_shot_mode="infer")
#
# Drop-in replacement for paper_replication.few_shot_inference().
# The only structural change: prompt construction uses
# build_few_shot_prompt_with_ti instead of build_few_shot_prompt_budgeted.
# Label scoring logic is identical to the baseline.
# ─────────────────────────────────────────────────────────────────────────────

def few_shot_inference_with_ti(
    model_name: str,
    train_texts: list[str],
    train_labels: list[str],
    test_texts: list[str],
    le,
    ti_mode: str = "none",
    num_shots: int = 3,
    max_prompt_tokens: int = 1024,
    prompt_style: str = "hybrid",
    label_diverse_shots: bool = True,
    use_quantization: bool = False,
) -> list[int]:
    """
    Few-shot inference with optional Thinking Intervention (TIbegin).

    When ti_mode != "none", every prompt ends with a partial assistant turn
    containing the intervention text. Label log-probabilities are then
    computed conditioned on this reasoning context, i.e.:

        score(label) = mean log P(label_token | prompt + thinking_prefix)

    This directly implements the TIbegin mechanism: the label selection is
    conditioned on the intervention reasoning injected before any model-
    generated tokens, giving the intervention maximum influence over the
    downstream prediction (Wu et al. §6.2, Figure 7).

    Args:
        model_name:          HuggingFace model name or local path.
        train_texts:         Training corpus texts (used for shot retrieval).
        train_labels:        Training corpus labels (string, not encoded).
        test_texts:          Test texts to classify.
        le:                  Fitted sklearn LabelEncoder.
        ti_mode:             One of TI_MODES. "none" reproduces the baseline.
        num_shots:           Number of few-shot examples per query.
        max_prompt_tokens:   Token budget for the full prompt.
        prompt_style:        "minimal" or "hybrid".
        label_diverse_shots: If True, maximise label coverage among shots.
        use_quantization:    Load model in 4-bit NF4 quantization.

    Returns:
        List of integer-encoded predictions (aligned with le.classes_).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    _ensure_tokenizer_pad_token(tokenizer)
    tokenizer.padding_side = "left"

    model_kwargs: dict = {"device_map": "auto", "torch_dtype": torch.bfloat16}
    if use_quantization:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()

    retriever = TfidfVectorizer(max_features=5000)
    train_matrix = retriever.fit_transform(train_texts)
    train_examples = list(zip(train_texts, train_labels))
    input_device = model.get_input_embeddings().weight.device

    # Pre-tokenize label strings for log-prob scoring (same as baseline).
    label_token_ids: dict[str, list[int]] = {}
    for label in le.classes_:
        ids = tokenizer(" " + label, add_special_tokens=False)["input_ids"]
        if len(ids) == 0:
            ids = tokenizer(label, add_special_tokens=False)["input_ids"]
        label_token_ids[label] = ids

    k = min(num_shots, len(train_examples))
    predictions: list[int] = []

    for text in tqdm(test_texts, desc=f"TI inference [{ti_mode}]"):
        query_vec = retriever.transform([text])
        sims = cosine_similarity(query_vec, train_matrix).ravel()
        top_idx = _select_topk_indices(
            sims, train_labels, k, label_diverse_shots=label_diverse_shots
        )
        examples = [train_examples[i] for i in top_idx]

        # ── Key difference from baseline: TI-aware prompt ─────────────────
        prompt = build_few_shot_prompt_with_ti(
            text,
            examples,
            list(le.classes_),
            tokenizer,
            ti_mode=ti_mode,
            model_name=model_name,
            max_prompt_tokens=max_prompt_tokens,
            prompt_style=prompt_style,
        )

        prompt_inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_prompt_tokens,
            add_special_tokens=False,
        ).to(input_device)

        prompt_ids = prompt_inputs["input_ids"]
        prompt_len = prompt_ids.shape[1]

        best_label: str | None = None
        best_score = float("-inf")

        # ── Label scoring — identical logic to paper_replication.py ───────
        with torch.no_grad():
            for label, ids in label_token_ids.items():
                if len(ids) == 0:
                    continue

                label_ids = torch.tensor(
                    ids, device=input_device, dtype=prompt_ids.dtype
                ).unsqueeze(0)
                full_ids = torch.cat([prompt_ids, label_ids], dim=1)
                full_mask = torch.ones_like(full_ids)

                logits = model(input_ids=full_ids, attention_mask=full_mask).logits

                # logits at position p+i-1 predict token at position p+i
                token_logits = logits[
                    :, prompt_len - 1 : prompt_len - 1 + len(ids), :
                ]
                log_probs = torch.log_softmax(token_logits, dim=-1)
                token_scores = log_probs.gather(
                    2, label_ids.unsqueeze(-1)
                ).squeeze(-1)
                # Normalize by label length (same as baseline).
                seq_score = token_scores.mean().item()

                if seq_score > best_score:
                    best_score = seq_score
                    best_label = label

        predictions.append(
            int(le.transform([best_label])[0]) if best_label is not None else -1
        )

    return predictions


# ─────────────────────────────────────────────────────────────────────────────
# 5. PROMPTED TEXT BUILDER WITH TI  (few_shot_mode="train")
#
# Variant of paper_replication.build_prompted_texts that uses
# build_few_shot_prompt_with_ti instead of build_few_shot_prompt_budgeted.
# The resulting prompted texts are passed to train_model() (from replication)
# so the classifier fine-tunes with TI-augmented inputs.
# ─────────────────────────────────────────────────────────────────────────────

def build_prompted_texts_with_ti(
    source_texts: list[str],
    source_labels: list[str],
    reference_texts: list[str],
    reference_labels: list[str],
    labels: list[str],
    tokenizer,
    model_name: str,
    ti_mode: str,
    num_shots: int = 3,
    max_prompt_tokens: int = 1024,
    prompt_style: str = "hybrid",
    label_diverse_shots: bool = True,
    exclude_self: bool = False,
) -> list[str]:
    """
    Builds prompted texts with TI prefix for fine-tuning (few_shot_mode="train").

    Note: TI in training mode is less theoretically grounded than in inference
    mode — the classification head learns from augmented inputs rather than
    directly conditioning label probabilities on the intervention. Use
    few_shot_mode="infer" for the principled TIbegin comparison.
    """
    retriever = TfidfVectorizer(max_features=5000)
    ref_matrix = retriever.fit_transform(reference_texts)
    src_matrix = retriever.transform(source_texts)

    prompted = []
    k = min(num_shots, len(reference_texts))

    for i in tqdm(range(len(source_texts)), desc="Building TI prompted texts", leave=False):
        sims = cosine_similarity(src_matrix[i], ref_matrix).ravel()
        if exclude_self and i < len(reference_texts):
            sims[i] = -1.0

        idx = _select_topk_indices(
            sims, reference_labels, k, label_diverse_shots=label_diverse_shots
        )
        examples = [(reference_texts[j], reference_labels[j]) for j in idx]

        prompt = build_few_shot_prompt_with_ti(
            source_texts[i],
            examples,
            labels,
            tokenizer,
            ti_mode=ti_mode,
            model_name=model_name,
            max_prompt_tokens=max_prompt_tokens,
            prompt_style=prompt_style,
        )
        prompted.append(prompt)

    return prompted


# ─────────────────────────────────────────────────────────────────────────────
# 6. EXPERIMENT TAG
# ─────────────────────────────────────────────────────────────────────────────

def build_ti_experiment_tag(args: argparse.Namespace) -> str:
    """Extends the baseline experiment tag with the TI mode suffix."""
    base = build_experiment_tag(args)
    return f"{base}__ti-{args.ti_mode}"


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    effective_max_len = 512 if args.paper_parity else args.max_len
    effective_prompt_budget = (
        min(args.prompt_budget_tokens, effective_max_len)
        if args.paper_parity
        else args.prompt_budget_tokens
    )

    experiment_dir = os.path.join(args.output_dir, build_ti_experiment_tag(args))
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"Experiment directory: {experiment_dir}")
    print(f"TI mode:              {args.ti_mode!r}")
    print(f"Few-shot mode:        {args.few_shot_mode!r}")
    print(f"Model:                {args.model_name}")
    print(f"Reasoning model:      {_model_supports_think_tags(args.model_name)}")

    print_gpu_utilization()
    train_df, dev_df, test_df = load_dataset(args.data_dir, args.dataset_type)

    train_texts, train_labels = preprocess(train_df)
    dev_texts,   dev_labels   = preprocess(dev_df)
    test_texts,  test_labels  = preprocess(test_df)

    y_train, y_dev, y_test, le = encode_labels(train_labels, dev_labels, test_labels)
    num_labels = int(max(np.max(y_train), np.max(y_dev), np.max(y_test)) + 1)

    start = time.time()

    # ── Path A: log-prob scoring inference ───────────────────────────────────
    if args.few_shot_mode == "infer":
        print("\n----- Running Few-Shot Inference with TI -----")

        y_pred = few_shot_inference_with_ti(
            model_name=args.model_name,
            train_texts=train_texts,
            train_labels=train_labels,
            test_texts=test_texts,
            le=le,
            ti_mode=args.ti_mode,
            num_shots=args.num_shots,
            max_prompt_tokens=effective_prompt_budget,
            prompt_style=args.prompt_style,
            label_diverse_shots=args.label_diverse_shots,
            use_quantization=args.use_quantization,
        )
        y_true = y_test

    # ── Path B: fine-tuning with TI-augmented prompts ────────────────────────
    elif args.few_shot_mode == "train":
        print("\n----- Running Few-Shot Training with TI -----")

        prompt_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        _ensure_tokenizer_pad_token(prompt_tokenizer)

        # Build TI-augmented prompted datasets for train / dev / test
        prompted_train = build_prompted_texts_with_ti(
            train_texts, train_labels,
            train_texts, train_labels,
            list(le.classes_), prompt_tokenizer,
            model_name=args.model_name,
            ti_mode=args.ti_mode,
            num_shots=args.num_shots,
            max_prompt_tokens=effective_prompt_budget,
            prompt_style=args.prompt_style,
            label_diverse_shots=args.label_diverse_shots,
            exclude_self=True,
        )
        prompted_dev = build_prompted_texts_with_ti(
            dev_texts, dev_labels,
            train_texts, train_labels,
            list(le.classes_), prompt_tokenizer,
            model_name=args.model_name,
            ti_mode=args.ti_mode,
            num_shots=args.num_shots,
            max_prompt_tokens=effective_prompt_budget,
            prompt_style=args.prompt_style,
            label_diverse_shots=args.label_diverse_shots,
            exclude_self=False,
        )
        prompted_test = build_prompted_texts_with_ti(
            test_texts, test_labels,
            train_texts, train_labels,
            list(le.classes_), prompt_tokenizer,
            model_name=args.model_name,
            ti_mode=args.ti_mode,
            num_shots=args.num_shots,
            max_prompt_tokens=effective_prompt_budget,
            prompt_style=args.prompt_style,
            label_diverse_shots=args.label_diverse_shots,
            exclude_self=False,
        )

        # Reuse train_model from paper_replication — no changes needed there
        trainer, _ = train_model(
            args.model_name,
            prompted_train,
            prompted_dev,
            y_train,
            y_dev,
            num_labels=num_labels,
            max_len=effective_max_len,
            use_quantization=args.use_quantization,
            use_lora=args.use_lora,
            use_torch_compile=args.use_torch_compile,
            seed=args.seed,
        )
        fs_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        _ensure_tokenizer_pad_token(fs_tokenizer)
        test_dataset = FallacyDataset(prompted_test, y_test, fs_tokenizer,
                                      max_len=effective_max_len)
        y_true, y_pred = evaluate(trainer, test_dataset)

    else:
        raise ValueError(f"Unknown few_shot_mode: {args.few_shot_mode!r}")

    elapsed = time.time() - start
    acc, p, r, f1 = compute_metrics(y_true, y_pred)

    print(f"\n{'='*50}")
    print(f"Results  ti_mode={args.ti_mode!r}  few_shot_mode={args.few_shot_mode!r}")
    print(f"  accuracy={acc:.4f}  precision={p:.4f}  recall={r:.4f}  f1={f1:.4f}")
    print(f"  time={elapsed:.1f}s")
    print(f"{'='*50}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    results_df = pd.DataFrame([{
        "ti_mode":        args.ti_mode,
        "few_shot_mode":  args.few_shot_mode,
        "model":          args.model_name,
        "dataset":        args.dataset_type,
        "num_shots":      args.num_shots,
        "prompt_style":   args.prompt_style,
        "accuracy":       acc,
        "precision":      p,
        "recall":         r,
        "f1":             f1,
        "time_s":         round(elapsed, 2),
    }])
    metrics_path = os.path.join(experiment_dir, "metrics.csv")
    results_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")

    preds_df = pd.DataFrame({
        "text":       test_texts,
        "true_label": [le.classes_[i] for i in y_true],
        "pred_label": [
            le.classes_[i] if 0 <= i < len(le.classes_) else "__UNPARSEABLE__"
            for i in y_pred
        ],
    })
    preds_path = os.path.join(experiment_dir, "predictions.csv")
    preds_df.to_csv(preds_path, index=False)
    print(f"Predictions saved to {preds_path}")

    plot_confusion(
        y_true, y_pred, le.classes_,
        save_path=os.path.join(experiment_dir, "confusion_matrix.png"),
    )
    print("Confusion matrix saved.")
    print_gpu_utilization()


# ─────────────────────────────────────────────────────────────────────────────
# 8. CLI
# All args mirror paper_replication.py exactly so results are directly
# comparable. The only additions are --ti_mode and --few_shot_mode.
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Few-shot fallacy detection with Thinking Intervention (Wu et al. 2025)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Mirrored from paper_replication.py ───────────────────────────────────
    parser.add_argument("--data_dir",             type=str,  default="Data")
    parser.add_argument("--dataset_type",         type=str,  default="edu",
                        choices=["climate", "edu", "all"])
    parser.add_argument("--model_name",           type=str,
                        default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output_dir",           type=str,  default="./Results")
    parser.add_argument("--max_len",              type=int,  default=512)
    parser.add_argument("--num_shots",            type=int,  default=3)
    parser.add_argument("--prompt_budget_tokens", type=int,  default=512)
    parser.add_argument("--seed",                 type=int,  default=42)
    parser.add_argument("--prompt_style",         type=str,  default="minimal",
                        choices=["minimal", "hybrid"])
    parser.add_argument("--label_diverse_shots",
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_quantization",
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_lora",
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_torch_compile",
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--paper_parity",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--few_shot_mode",        type=str,  default="infer",
                        choices=["infer", "train"],
                        help=(
                            "'infer': log-prob scoring — TI is most principled here. "
                            "'train': fine-tune classifier with TI-augmented prompts."
                        ))

    # ── TI-specific ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--ti_mode", type=str, default="none",
        choices=list(TI_MODES),
        help=(
            "Thinking Intervention mode. "
            "'none' exactly replicates paper_replication.py few-shot inference. "
            "'structure_focus' is the recommended starting point — it directly "
            "targets the equivocation/intentional failure modes reported in Teo et al."
        ),
    )

    args = parser.parse_args()

    # training_mode is required by build_experiment_tag from paper_replication.
    # We fix it to "few-shot" since this script only does few-shot.
    args.training_mode = "few-shot"

    main(args)
