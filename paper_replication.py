"""
Replicates LLM for logical fallacy detection paper

FOR NOW DO NOT PUSH MODIFICATIONS TO THIS FILE. THIS IS A WIP AND WANT TO KEEP IT SO WE CAN TRY TO HAVE SOMETHING AS CLOSE TO THE LLM FOR FALLACY PAPER AS POSSIBLE
"""
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, BitsAndBytesConfig
from transformers import set_seed
from peft import get_peft_model, LoraConfig, TaskType
from pynvml import *
from tqdm import tqdm

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")
    print(f"GPU memory free: {info.free//1024**2} MB.")

############################################
# 1. DATA LOADING
############################################

def load_split(base_path, dataset_type, split):
    """
    dataset_type: climate | edu | all
    split: train | dev | test
    """

    if dataset_type == "climate":
        path = os.path.join(base_path, "climate_data", f"logic_climate_{split}.csv")

    elif dataset_type == "edu":
        path = os.path.join(base_path, "edu_data", f"logic_edu_{split}.csv")

    elif dataset_type == "all":
        path = os.path.join(base_path, f"all_logic_{split}.csv")

    else:
        raise ValueError("dataset_type must be: climate, edu, or all")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)
    df = df.dropna()

    return df


def load_dataset(base_path, dataset_type):
    train_df = load_split(base_path, dataset_type, "train")
    dev_df   = load_split(base_path, dataset_type, "dev")
    test_df  = load_split(base_path, dataset_type, "test")

    return train_df, dev_df, test_df

############################################
# 2. PREPROCESSING (matches paper)
############################################

def preprocess(df):
    texts = df["source_article"].tolist()
    labels = df["logical_fallacies"].tolist()

    return texts, labels


def encode_labels(train_labels, dev_labels, test_labels):
    le = LabelEncoder()

    y_train = le.fit_transform(train_labels)
    y_dev   = le.transform(dev_labels)
    y_test  = le.transform(test_labels)

    return y_train, y_dev, y_test, le


def _validate_label_range(y, split_name, num_labels):
    y = np.asarray(y)
    if y.size == 0:
        raise ValueError(f"{split_name} labels are empty.")
    y_min = int(np.min(y))
    y_max = int(np.max(y))
    if y_min < 0 or y_max >= num_labels:
        raise ValueError(
            f"Label range error in {split_name}: min={y_min}, max={y_max}, num_labels={num_labels}."
        )


def _get_classifier_out_features(model):
    """Best-effort lookup of classifier output size across HF/PEFT wrappers."""
    candidate_suffixes = ("score", "classifier", "classification_head")
    for module_name, module in model.named_modules():
        if module_name.endswith(candidate_suffixes) and hasattr(module, "out_features"):
            return int(module.out_features), module_name
    return None, None


def _validate_model_label_compat(model, num_labels):
    cfg_labels = getattr(model.config, "num_labels", None)
    if cfg_labels is not None and int(cfg_labels) != int(num_labels):
        raise ValueError(
            f"Model config num_labels mismatch: model.config.num_labels={cfg_labels}, expected={num_labels}."
        )

    head_out, head_name = _get_classifier_out_features(model)
    if head_out is not None and int(head_out) != int(num_labels):
        raise ValueError(
            f"Classifier head mismatch at '{head_name}': out_features={head_out}, expected={num_labels}."
        )
############################################
# 3. TF-IDF BASELINE (paper baseline)
############################################

def tfidf_pipeline(X_train, X_test):
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    return X_train_vec, X_test_vec


############################################
# 4. DATASET CLASS FOR HF TRAINING
############################################

class FallacyDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_len
        )
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


############################################
# 5. FEW-SHOT PROMPTING
############################################

def build_few_shot_prompt(text, examples, labels, tokenizer=None, prompt_style="hybrid"):
    """
    Builds a few-shot prompt.
    prompt_style:
      - minimal: close to paper wording
      - hybrid: concise constraints + label options
    """
    system_prompt = "You are a helpful assistant."

    if prompt_style == "minimal":
        user_prompt = "Classify the logical fallacy:\n\n"
        for ex_text, ex_label in examples:
            user_prompt += f"Text: {ex_text}\nLabel: {ex_label}\n\n"
        user_prompt += f"Text: {text}\nLabel:"
    else:
        # Balanced template: explicit label constraints without overloading instructions.
        user_prompt = (
            "Classify the logical fallacy.\n"
            "Return exactly one label from the options below.\n"
            "Output only the label text.\n\n"
            f"Options: {', '.join(labels)}\n\n"
        )
        for ex_text, ex_label in examples:
            user_prompt += f"Text: {ex_text}\nLabel: {ex_label}\n\n"
        user_prompt += f"Text: {text}\nLabel:"

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # Fallback
    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    prompt += f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
    prompt += f"<|im_start|>assistant\n"
    return prompt

def _truncate_text_to_tokens(text, tokenizer, max_tokens):
    if max_tokens <= 0:
        return ""
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    token_ids = token_ids[:max_tokens]
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def _select_topk_indices(similarities, reference_labels, k, label_diverse_shots=True):
    ranked = np.argsort(similarities)[::-1]
    if not label_diverse_shots:
        return ranked[:k].tolist()

    selected = []
    seen_labels = set()

    # First pass: maximize label coverage among top-relevant examples.
    for idx in ranked:
        label = reference_labels[idx]
        if label not in seen_labels:
            selected.append(int(idx))
            seen_labels.add(label)
            if len(selected) == k:
                return selected

    # Second pass: fill remaining slots by relevance.
    for idx in ranked:
        idx = int(idx)
        if idx not in selected:
            selected.append(idx)
            if len(selected) == k:
                break

    return selected


def build_few_shot_prompt_budgeted(
    text,
    examples,
    labels,
    tokenizer,
    max_prompt_tokens=1024,
    prompt_style="hybrid",
):
    """Build a prompt while preserving shots + query under a token budget."""
    k = max(1, len(examples))
    # Reserve room for template tokens, labels, and delimiters.
    available = max(160, max_prompt_tokens - 96)
    example_budget = max(32, int((available * 0.55) / k))
    query_budget = max(64, available - (example_budget * k))

    trimmed_examples = [
        (_truncate_text_to_tokens(ex_text, tokenizer, example_budget), ex_label)
        for ex_text, ex_label in examples
    ]
    trimmed_text = _truncate_text_to_tokens(text, tokenizer, query_budget)
    prompt = build_few_shot_prompt(
        trimmed_text,
        trimmed_examples,
        labels=labels,
        tokenizer=tokenizer,
        prompt_style=prompt_style,
    )

    # If we still exceed budget, shrink iteratively.
    cur_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    while cur_len > max_prompt_tokens and (example_budget > 16 or query_budget > 32):
        example_budget = max(16, int(example_budget * 0.85))
        query_budget = max(32, int(query_budget * 0.9))
        trimmed_examples = [
            (_truncate_text_to_tokens(ex_text, tokenizer, example_budget), ex_label)
            for ex_text, ex_label in examples
        ]
        trimmed_text = _truncate_text_to_tokens(text, tokenizer, query_budget)
        prompt = build_few_shot_prompt(
            trimmed_text,
            trimmed_examples,
            labels=labels,
            tokenizer=tokenizer,
            prompt_style=prompt_style,
        )
        cur_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])

    return prompt


def build_prompted_texts(
    source_texts,
    source_labels,
    reference_texts,
    reference_labels,
    labels,
    tokenizer,
    num_shots=3,
    max_prompt_tokens=1024,
    prompt_style="hybrid",
    label_diverse_shots=True,
    exclude_self=False,
):
    """Create prompted instances by retrieving top-k relevant examples via TF-IDF similarity."""
    retriever = TfidfVectorizer(max_features=5000)
    ref_matrix = retriever.fit_transform(reference_texts)
    src_matrix = retriever.transform(source_texts)

    prompted = []
    k = min(num_shots, len(reference_texts))
    for i in tqdm(range(len(source_texts)), desc="Building prompted texts", leave=False):
        sims = cosine_similarity(src_matrix[i], ref_matrix).ravel()
        if exclude_self and i < len(reference_texts):
            sims[i] = -1.0

        idx = _select_topk_indices(sims, reference_labels, k, label_diverse_shots=label_diverse_shots)
        examples = [(reference_texts[j], reference_labels[j]) for j in idx]
        prompt = build_few_shot_prompt_budgeted(
            source_texts[i],
            examples,
            labels,
            tokenizer,
            max_prompt_tokens=max_prompt_tokens,
            prompt_style=prompt_style,
        )
        prompted.append(prompt)

    return prompted


############################################
# 6. FEW-SHOT INFERENCE
############################################

def few_shot_inference(
    model_name,
    train_texts,
    train_labels,
    test_texts,
    le,
    num_shots=3,
    max_prompt_tokens=1024,
    prompt_style="hybrid",
    label_diverse_shots=True,
    use_quantization=True,
):
    """
    Performs few-shot inference on the test set using a causal LM.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Important for generative models

    model_kwargs = {
        "device_map": "auto",
        "dtype": torch.bfloat16,
    }
    if use_quantization:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()

    # Build a lightweight relevance retriever to pick the top-k most similar shots.
    retriever = TfidfVectorizer(max_features=5000)
    train_matrix = retriever.fit_transform(train_texts)
    train_examples = list(zip(train_texts, train_labels))
    input_device = model.get_input_embeddings().weight.device

    # Pre-tokenize label strings so we can score candidate labels directly.
    label_token_ids = {}
    for label in le.classes_:
        ids = tokenizer(" " + label, add_special_tokens=False)["input_ids"]
        if len(ids) == 0:
            ids = tokenizer(label, add_special_tokens=False)["input_ids"]
        label_token_ids[label] = ids

    k = min(num_shots, len(train_examples))

    predictions = []
    for text in tqdm(test_texts, desc="Few-shot inference"):
        query_vec = retriever.transform([text])
        sims = cosine_similarity(query_vec, train_matrix).ravel()
        top_idx = _select_topk_indices(sims, train_labels, k, label_diverse_shots=label_diverse_shots)
        examples = [train_examples[i] for i in top_idx]

        examples_with_text_labels = [(ex_text, ex_label) for ex_text, ex_label in examples]

        prompt = build_few_shot_prompt_budgeted(
            text,
            examples_with_text_labels,
            list(le.classes_),
            tokenizer,
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

        best_label = None
        best_score = float("-inf")

        # Score each possible label continuation and pick the highest-likelihood label.
        with torch.no_grad():
            for label, ids in label_token_ids.items():
                if len(ids) == 0:
                    continue

                label_ids = torch.tensor(ids, device=input_device, dtype=prompt_ids.dtype).unsqueeze(0)
                full_ids = torch.cat([prompt_ids, label_ids], dim=1)
                full_mask = torch.ones_like(full_ids)

                logits = model(input_ids=full_ids, attention_mask=full_mask).logits

                # For each label token t_i at position p+i, use logits at position p+i-1.
                token_logits = logits[:, prompt_len - 1:prompt_len - 1 + len(ids), :]
                log_probs = torch.log_softmax(token_logits, dim=-1)
                token_scores = log_probs.gather(2, label_ids.unsqueeze(-1)).squeeze(-1)
                # Normalize by label length so multi-token labels are not penalized.
                seq_score = token_scores.mean().item()

                if seq_score > best_score:
                    best_score = seq_score
                    best_label = label

        if best_label is None:
            predictions.append(-1)
        else:
            predictions.append(int(le.transform([best_label])[0]))

    return predictions


############################################
# 7. METRICS
############################################

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    return acc, p, r, f1


############################################
# 7. CONFUSION MATRIX
############################################

def plot_confusion(y_true, y_pred, label_names, save_path="cm.png"):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=label_names,
                yticklabels=label_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(save_path)
    plt.close()

def build_experiment_tag(args: argparse.Namespace) -> str:
    """Build a human-readable tag that encodes the experiment configuration.

    Example output:
        data-all_logic__bert-base-uncased+roberta-base__ep3_bs16_lr2e-05_ml256
    """
    # ── Data tag: derive from train filename (drop _train.csv suffix) ────
    data_tag = args.dataset_type

    # ── Model tag: short names joined with '+' ──────────────────────────
    model_tag = args.model_name.replace("/", "_")

    # ── Hyper-param tag ─────────────────────────────────────────────────
    effective_max_len = 512 if args.paper_parity else args.max_len
    hp_tag = f"ep5_bs1_lr2e-05_ml{effective_max_len}"

    mode_tag = args.training_mode
    if args.training_mode in ["few-shot", "all"]:
        mode_tag += (
            f"__fsmode-{args.few_shot_mode}"
            f"_shots-{args.num_shots}"
            f"_pb-{args.prompt_budget_tokens}"
            f"_pstyle-{args.prompt_style}"
            f"_div-{str(args.label_diverse_shots).lower()}"
        )
    mode_tag += (
        f"__parity-{str(args.paper_parity).lower()}"
        f"__quant-{str(args.use_quantization).lower()}"
        f"__lora-{str(args.use_lora).lower()}"
    )

    return f"data-{data_tag}__{model_tag}__{hp_tag}__{mode_tag}"

############################################
# 8. TRAIN LLM CLASSIFIER
############################################

def train_model(
    model_name,
    train_texts,
    val_texts,
    train_labels,
    val_labels,
    num_labels,
    max_len=512,
    use_quantization=True,
    use_lora=True,
    use_torch_compile=False,
    seed=42,
):

    train_labels = np.asarray(train_labels, dtype=np.int64)
    val_labels = np.asarray(val_labels, dtype=np.int64)
    _validate_label_range(train_labels, "train(train_model)", num_labels)
    _validate_label_range(val_labels, "val(train_model)", num_labels)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "num_labels": num_labels,
    }
    if use_quantization:
        # For quantized runs, keep model sharded automatically.
        model_kwargs["device_map"] = "auto"
        model_kwargs["dtype"] = torch.bfloat16
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForSequenceClassification.from_pretrained(model_name, **model_kwargs)
    _validate_model_label_compat(model, num_labels)

    # Paper reports resizing token embeddings to tokenizer vocabulary size.
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    if use_lora:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=16,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        _validate_model_label_compat(model, num_labels)


    train_dataset = FallacyDataset(train_texts, train_labels, tokenizer, max_len=max_len)
    val_dataset   = FallacyDataset(val_texts, val_labels, tokenizer, max_len=max_len)

    training_args = TrainingArguments(
        output_dir="./checkpoints",
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_dir="./logs",
        seed=seed,
        data_seed=seed,
        torch_compile=use_torch_compile,
        optim="paged_adamw_8bit" if use_quantization else "adamw_torch",
        max_grad_norm=0.3,
        warmup_ratio=0.03
    )

    def compute_metrics_hf(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)

        acc, p, r, f1 = compute_metrics(labels, preds)

        return {
            "accuracy": acc,
            "precision": p,
            "recall": r,
            "f1": f1
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_hf,
    )

    try:
        trainer.train()
    except Exception as exc:
        train_min = int(np.min(train_labels))
        train_max = int(np.max(train_labels))
        val_min = int(np.min(val_labels))
        val_max = int(np.max(val_labels))
        print(
            "Training failed diagnostics: "
            f"num_labels={num_labels}, "
            f"train_label_range=[{train_min}, {train_max}], "
            f"val_label_range=[{val_min}, {val_max}]"
        )
        raise RuntimeError(
            "Training failed. This is often a label/head mismatch or CUDA assert. "
            "If this persists, run with CUDA_LAUNCH_BLOCKING=1 for a synchronous traceback."
        ) from exc

    return trainer, tokenizer

############################################
# 9. EVALUATION
############################################

def evaluate(trainer, dataset):
    preds = trainer.predict(dataset)
    y_pred = np.argmax(preds.predictions, axis=1)
    y_true = preds.label_ids

    return y_true, y_pred


############################################
# 10. MAIN
############################################

def main(args):

    set_seed(args.seed)

    effective_max_len = 512 if args.paper_parity else args.max_len
    effective_prompt_budget = min(args.prompt_budget_tokens, effective_max_len) if args.paper_parity else args.prompt_budget_tokens

    experiment_dir = os.path.join(args.output_dir, build_experiment_tag(args))
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"Experiment directory: {experiment_dir}")

    print_gpu_utilization()
    train_df, dev_df, test_df = load_dataset(args.data_dir, args.dataset_type)

    train_texts, train_labels = preprocess(train_df)
    dev_texts, dev_labels = preprocess(dev_df)
    test_texts, test_labels = preprocess(test_df)

    y_train, y_dev, y_test, le = encode_labels(train_labels, dev_labels, test_labels)
    # Derive label count from encoded targets to prevent classifier head mismatch.
    num_labels = int(max(np.max(y_train), np.max(y_dev), np.max(y_test)) + 1)
    _validate_label_range(y_train, "train", num_labels)
    _validate_label_range(y_dev, "dev", num_labels)
    _validate_label_range(y_test, "test", num_labels)

    if num_labels != len(le.classes_):
        print(
            f"Warning: num_labels from encoded targets ({num_labels}) differs from LabelEncoder classes ({len(le.classes_)})."
        )

    print_gpu_utilization()
    ########################################
    # BASELINE TF-IDF
    ########################################
    X_train_vec, X_test_vec = tfidf_pipeline(train_texts, test_texts)

    print("TF-IDF baseline prepared.")
    print_gpu_utilization()

    all_results = []

    # --- Baseline Run ---
    if args.training_mode in ['baseline', 'all']:
        print("\n----- Running Baseline Training -----")
        start_time = time.time()
        
        trainer, tokenizer = train_model(
            args.model_name,
            train_texts,
            dev_texts,
            y_train,
            y_dev,
            num_labels=num_labels,
            max_len=effective_max_len,
            use_quantization=args.use_quantization,
            use_lora=args.use_lora,
            use_torch_compile=args.use_torch_compile,
            seed=args.seed,
        )
        
        test_dataset = FallacyDataset(test_texts, y_test, tokenizer, max_len=effective_max_len)
        y_true, y_pred = evaluate(trainer, test_dataset)
        
        training_time = time.time() - start_time
        
        acc, p, r, f1 = compute_metrics(y_true, y_pred)
        
        results = {
            "mode": "baseline",
            "model": args.model_name,
            "dataset": args.dataset_type,
            "accuracy": acc,
            "precision": p,
            "recall": r,
            "f1": f1,
            "train_time_s": training_time,
        }
        all_results.append(results)

        # Save predictions and confusion matrix for baseline
        preds_df = pd.DataFrame({
            "text": test_texts,
            "true_label": [le.classes_[i] for i in y_true],
            "pred_label": [le.classes_[i] for i in y_pred],
        })
        preds_csv = os.path.join(experiment_dir, "predictions_baseline.csv")
        preds_df.to_csv(preds_csv, index=False)
        print(f"Baseline predictions saved to {preds_csv}")

        plot_confusion(y_true, y_pred, le.classes_, save_path=os.path.join(experiment_dir, "confusion_matrix_baseline.png"))
        print("Baseline confusion matrix saved.")


    # --- Few-Shot Run ---
    if args.training_mode in ['few-shot', 'all']:
        print(f"\n----- Running Few-Shot ({args.few_shot_mode}) -----")
        start_time = time.time()

        if args.few_shot_mode == "infer":
            y_pred = few_shot_inference(
                args.model_name,
                train_texts,
                train_labels,
                test_texts,
                le,
                num_shots=args.num_shots,
                max_prompt_tokens=effective_prompt_budget,
                prompt_style=args.prompt_style,
                label_diverse_shots=args.label_diverse_shots,
                use_quantization=args.use_quantization,
            )
            y_true = y_test
        else:
            prompt_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            prompted_train = build_prompted_texts(
                train_texts,
                train_labels,
                train_texts,
                train_labels,
                list(le.classes_),
                prompt_tokenizer,
                num_shots=args.num_shots,
                max_prompt_tokens=effective_prompt_budget,
                prompt_style=args.prompt_style,
                label_diverse_shots=args.label_diverse_shots,
                exclude_self=True,
            )
            prompted_dev = build_prompted_texts(
                dev_texts,
                dev_labels,
                train_texts,
                train_labels,
                list(le.classes_),
                prompt_tokenizer,
                num_shots=args.num_shots,
                max_prompt_tokens=effective_prompt_budget,
                prompt_style=args.prompt_style,
                label_diverse_shots=args.label_diverse_shots,
                exclude_self=False,
            )
            prompted_test = build_prompted_texts(
                test_texts,
                test_labels,
                train_texts,
                train_labels,
                list(le.classes_),
                prompt_tokenizer,
                num_shots=args.num_shots,
                max_prompt_tokens=effective_prompt_budget,
                prompt_style=args.prompt_style,
                label_diverse_shots=args.label_diverse_shots,
                exclude_self=False,
            )

            fs_trainer, fs_tokenizer = train_model(
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
            fs_test_dataset = FallacyDataset(prompted_test, y_test, fs_tokenizer, max_len=effective_max_len)
            y_true, y_pred = evaluate(fs_trainer, fs_test_dataset)
        
        inference_time = time.time() - start_time

        acc, p, r, f1 = compute_metrics(y_true, y_pred)

        results = {
            "mode": "few-shot",
            "model": args.model_name,
            "dataset": args.dataset_type,
            "accuracy": acc,
            "precision": p,
            "recall": r,
            "f1": f1,
            "train_time_s": inference_time,
        }
        all_results.append(results)

        # Save predictions and confusion matrix for few-shot
        preds_df = pd.DataFrame({
            "text": test_texts,
            "true_label": [le.classes_[i] for i in y_true],
            "pred_label": [le.classes_[i] for i in y_pred],
        })
        preds_csv = os.path.join(experiment_dir, "predictions_few-shot.csv")
        preds_df.to_csv(preds_csv, index=False)
        print(f"Few-shot predictions saved to {preds_csv}")

        plot_confusion(y_true, y_pred, le.classes_, save_path=os.path.join(experiment_dir, "confusion_matrix_few-shot.png"))
        print("Few-shot confusion matrix saved.")

    if not all_results:
        raise ValueError("Invalid training_mode. Choose 'baseline', 'few-shot', or 'all'.")

    # Save combined metrics
    metrics_df = pd.DataFrame(all_results)
    metrics_csv = os.path.join(experiment_dir, "metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"\nCombined metrics saved to {metrics_csv}")

    print("\n===== COMBINED RESULTS =====")
    print(metrics_df)

    print_gpu_utilization()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="Data")
    parser.add_argument(
        "--dataset_type",
        type=str,
        choices=["climate", "edu", "all"],
        default="edu"
    )
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")#default="Qwen/Qwen2-7B-Instruct")
    parser.add_argument("--output_dir", type=str, default="./Results")
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--num_shots", type=int, default=3)
    parser.add_argument("--prompt_budget_tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use_torch_compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable torch.compile in TrainingArguments.",
    )
    parser.add_argument(
        "--use_quantization",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use 4-bit quantization for model loading.",
    )
    parser.add_argument(
        "--use_lora",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use LoRA adapters during sequence-classification training.",
    )
    parser.add_argument(
        "--paper_parity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable strict paper parity settings (forces effective max length to 512).",
    )
    parser.add_argument(
        "--label_diverse_shots",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Select few-shot examples to maximize label diversity among top-relevant candidates.",
    )
    parser.add_argument(
        "--prompt_style",
        type=str,
        choices=["minimal", "hybrid"],
        default="minimal",
        help="Few-shot prompt template style.",
    )
    parser.add_argument(
        "--few_shot_mode",
        type=str,
        choices=["train", "infer"],
        default="train",
        help="Few-shot implementation: 'train' builds prompted datasets and trains; 'infer' does in-context scoring only.",
    )
    parser.add_argument(
        "--training_mode",
        type=str,
        choices=["baseline", "few-shot", "all"],
        default="few-shot",
        help="Training mode: 'baseline' for fine-tuning, 'few-shot' for few-shot inference. Use 'all' to run both sequentially."
    )

    args = parser.parse_args()

    main(args)