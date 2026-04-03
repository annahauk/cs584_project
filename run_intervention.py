import os
import argparse
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import ollama
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
INTERVENTION_TEXT = """
I should carefully analyze the argument.
I should identify whether it contains a logical fallacy.
I should consider known fallacy types like ad hominem, strawman, false cause.
I should justify my reasoning step by step.
"""

FALLACY_LABELS = ['appeal to emotion', 
                  'false causality', 
                  'ad populum', 
                  'circular reasoning', 
                  'fallacy of relevance', 
                  'faulty generalization', 
                  'ad hominem', 
                  'fallacy of extension', 
                  'equivocation', 
                  'fallacy of logic', 
                  'fallacy of credibility', 
                  'intentional', 
                  'false dilemma']

MODEL_NAME = ""

# =========================
# DATA LOADING
# =========================
def load_dataset(base_path, subset, split):
    if subset == "all":
        path = os.path.join(base_path, f"all_logic_{split}.csv")
    elif subset == "edu":
        path = os.path.join(base_path, "edu_data", f"logic_edu_{split}.csv")
    elif subset == "climate":
        path = os.path.join(base_path, "climate_data", f"logic_climate_{split}.csv")
    else:
        raise ValueError("Invalid subset")

    df = pd.read_csv(path)
    return df

# =========================
# PROMPT
# =========================
def build_prompt(argument):
    labels_str = ", ".join(FALLACY_LABELS)
    return f"""
Identify the logical fallacy in the argument.

You MUST choose one label from this list:
[{labels_str}]

Argument:
"{argument}"

Answer format:
Label: <one of the labels>
Explanation: <reasoning>

<think>
"""

# =========================
# BASELINE
# =========================
def baseline_inference(model, tokenizer, argument):
    prompt = build_prompt(argument)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# =========================
# TRANSFORMERS INTERVENTION
# =========================
def get_label_token_ids(tokenizer):
    label_token_ids = []
    for label in FALLACY_LABELS:
        ids = tokenizer(label, add_special_tokens=False).input_ids
        label_token_ids.extend(ids)
    return list(set(label_token_ids))

class LabelConstraintProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids, tokenizer, trigger_phrase="Label:"):
        self.allowed_token_ids = allowed_token_ids
        self.tokenizer = tokenizer
        self.trigger_phrase = trigger_phrase
        self.active = False

    def __call__(self, input_ids, scores):
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

        if self.trigger_phrase in text:
            self.active = True

        if self.active:
            mask = torch.full_like(scores, float("-inf"))
            mask[:, self.allowed_token_ids] = 0
            return mask

        return scores

class InterventionProcessor(LogitsProcessor):
    def __init__(self, intervention_ids, trigger_ids):
        self.intervention_ids = intervention_ids
        self.trigger_ids = trigger_ids
        self.intervened = False
        self.inject_index = 0

    def __call__(self, input_ids, scores):
        # Check trigger
        if not self.intervened and any(t in input_ids[0].tolist() for t in self.trigger_ids):
            self.intervened = True

        # If intervening, force tokens sequentially
        if self.intervened and self.inject_index < len(self.intervention_ids):
            forced_token = self.intervention_ids[self.inject_index]
            self.inject_index += 1

            mask = torch.full_like(scores, float("-inf"))
            mask[:, forced_token] = 0
            return mask

        return scores


def transformers_intervention(model, tokenizer, argument):
    prompt = build_prompt(argument)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    intervention_ids = tokenizer(INTERVENTION_TEXT, add_special_tokens=False).input_ids
    trigger_ids = tokenizer("<think>", add_special_tokens=False).input_ids
    label_ids = get_label_token_ids(tokenizer)

    processor = InterventionProcessor(intervention_ids, trigger_ids)
    constraint = LabelConstraintProcessor(label_ids, tokenizer)

    logits_processor = LogitsProcessorList([processor, constraint])

    outputs = model.generate(
        input_ids,
        max_new_tokens=128,
        logits_processor=logits_processor
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# =========================
# OLLAMA INTERVENTION
# =========================
def ollama_intervention(argument, model_name="qwen:latest"):

    prompt = build_prompt(argument)

    stream = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    generated = ""
    intervened = False

    for chunk in stream:
        token = chunk["message"]["content"]
        generated += token

        if not intervened and "<think>" in generated:
            intervened = True
            break

    if intervened:
        new_prompt = generated + "\n" + INTERVENTION_TEXT

        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": new_prompt}],
        )

        return response["message"]["content"]

    return generated


# =========================
# SIMPLE LABEL EXTRACTION
# =========================
def extract_label(output):
    for label in FALLACY_LABELS:
        if f"label: {label}" in output.lower():
            return label
    return "unknown"


# =========================
# RUN EXPERIMENT
# =========================
def run_experiment(args):
    df = load_dataset(args.data_path, args.subset, args.split)

    print(f"Loaded {len(df)} samples")

    # Load HF model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype="auto"
    )

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        argument = row["source_article"]
        true_label = row["logical_fallacies"]

        if args.method == "baseline":
            output = baseline_inference(model, tokenizer, argument)

        elif args.method == "transformers":
            output = transformers_intervention(model, tokenizer, argument)

        elif args.method == "ollama":
            output = ollama_intervention(argument, args.ollama_model)

        else:
            raise ValueError("Invalid method")

        pred_label = extract_label(output)

        results.append({
            "argument": argument,
            "true": true_label,
            "pred": pred_label,
            "output": output
        })

    results_df = pd.DataFrame(results)

    acc = accuracy_score(results_df["true"], results_df["pred"])
    f1 = f1_score(results_df["true"], results_df["pred"], average="macro")

    print(f"\nResults ({args.method}):")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1: {f1:.4f}")

    results_df.to_csv(f"results_{args.method}_{args.subset}_{args.split}.csv", index=False)
    return acc, f1

def plot_results(results_dict, save_path="Results/"):
    methods = list(results_dict.keys())
    accs = [results_dict[m]["accuracy"] for m in methods]
    f1s = [results_dict[m]["f1"] for m in methods]

    x = range(len(methods))

    plt.figure()
    plt.bar(x, accs)
    plt.xticks(x, methods)
    plt.title("Accuracy by Method")
    plt.savefig(save_path + "accuracy_plot.png")

    plt.figure()
    plt.bar(x, f1s)
    plt.xticks(x, methods)
    plt.title("F1 Score by Method")
    plt.savefig(save_path + "f1_plot.png")

def run_all_methods(args):
    methods = ["baseline", "ollama", "transformers"]
    results_summary = {}

    for method in methods:
        args.method = method
        acc, f1 = run_experiment(args)
        results_summary[method] = {"accuracy": acc, "f1": f1}

    plot_results(results_summary)

# =========================
# CLI
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--subset", choices=["all", "edu", "climate"], default="all")
    parser.add_argument("--split", choices=["train", "dev", "test"], default="test")
    parser.add_argument("--method", choices=["baseline", "ollama", "transformers"], required=True)
    parser.add_argument("--hf_model", type=str, default="Qwen/Qwen2-7B-Instruct")
    parser.add_argument("--ollama_model", type=str, default="qwen:latest")

    args = parser.parse_args()
    MODEL_NAME = args.hf_model  

    run_experiment(args)