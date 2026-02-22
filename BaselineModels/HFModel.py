from __future__ import annotations
import json
import os
from typing import Any
import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from utils.TextClassificationDataset import TextClassificationDataset

class HFModel:
    """Reusable non-LLM Hugging Face text classification model wrapper."""

    def __init__(
        self,
        model_name: str,
        text_column: str = "source_article",
        label_column: str = "logical_fallacies",
        max_length: int = 256,
        batch_size: int = 16,
        num_train_epochs: int = 3,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        output_dir: str = "outputs",
        seed: int = 42,
    ):
        self.model_name = model_name
        self.text_column = text_column
        self.label_column = label_column
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_train_epochs = num_train_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.output_dir = output_dir
        self.seed = seed

        self.model = None
        self.tokenizer = None
        self.trainer = None

        self.train_texts: list[str] = []
        self.dev_texts: list[str] = []
        self.test_texts: list[str] = []
        self.train_labels: list[int] | None = None
        self.dev_labels: list[int] | None = None
        self.test_labels: list[int] | None = None

        self.label2id: dict[str, int] = {}
        self.id2label: dict[int, str] = {}

    def load_train_dev_test_data(self, train_path: str, dev_path: str, test_path: str):
        """Load train/dev/test CSV files using fixed project schema columns."""
        train_df = pd.read_csv(train_path)
        dev_df = pd.read_csv(dev_path)
        test_df = pd.read_csv(test_path)

        if self.text_column not in train_df.columns:
            raise ValueError(f"Text column '{self.text_column}' not found in train data")
        if self.label_column not in train_df.columns:
            raise ValueError(f"Label column '{self.label_column}' not found in train data")

        self.train_texts = train_df[self.text_column].fillna("").astype(str).tolist()
        self.dev_texts = (
            dev_df[self.text_column].fillna("").astype(str).tolist()
            if self.text_column in dev_df.columns
            else []
        )
        self.test_texts = (
            test_df[self.text_column].fillna("").astype(str).tolist()
            if self.text_column in test_df.columns
            else []
        )

        train_raw_labels = train_df[self.label_column].fillna("").astype(str).tolist()
        unique_labels = sorted(set(train_raw_labels))
        self.label2id = {label: index for index, label in enumerate(unique_labels)}
        self.id2label = {index: label for label, index in self.label2id.items()}

        self.train_labels = [self.label2id[label] for label in train_raw_labels]

        if self.label_column in dev_df.columns:
            dev_raw_labels = dev_df[self.label_column].fillna("").astype(str).tolist()
            self.dev_labels = [self.label2id[label] for label in dev_raw_labels]
        else:
            self.dev_labels = None

        if self.label_column in test_df.columns:
            test_raw_labels = test_df[self.label_column].fillna("").astype(str).tolist()
            self.test_labels = [self.label2id[label] for label in test_raw_labels]
        else:
            self.test_labels = None

    def load_model(self):
        """Load tokenizer + sequence classification model with current label mappings."""
        label_map_path = os.path.join(self.model_name, "label_mappings.json")
        if os.path.exists(label_map_path) and not self.label2id:
            with open(label_map_path, "r", encoding="utf-8") as map_file:
                mapping = json.load(map_file)
            self.label2id = {str(k): int(v) for k, v in mapping.get("label2id", {}).items()}
            self.id2label = {int(k): str(v) for k, v in mapping.get("id2label", {}).items()}

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

        model_kwargs = {}
        if self.label2id:
            model_kwargs["num_labels"] = len(self.label2id)
            model_kwargs["label2id"] = self.label2id
            model_kwargs["id2label"] = self.id2label

        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, **model_kwargs)

        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

        return self.model

    def save_model(self, save_path: str):
        """Save model/tokenizer and metadata to disk."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model/tokenizer are not loaded. Call load_model() or train() first.")

        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        with open(os.path.join(save_path, "label_mappings.json"), "w", encoding="utf-8") as map_file:
            json.dump(
                {
                    "label2id": self.label2id,
                    "id2label": {str(k): v for k, v in self.id2label.items()},
                    "text_column": self.text_column,
                    "label_column": self.label_column,
                },
                map_file,
                indent=2,
            )

    def _build_dataset(self, texts: list[str], labels: list[int] | None) -> TextClassificationDataset:
        """Tokenize without padding; padding is handled dynamically per-batch
        by DataCollatorWithPadding in the Trainer, which is much more memory
        efficient than padding every sample to max_length upfront."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not loaded. Call load_model() first.")

        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=False,        # dynamic padding via DataCollator
            max_length=self.max_length,
        )
        return TextClassificationDataset(encodings=encodings, labels=labels)

    @staticmethod
    def _compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = float((predictions == labels).mean())
        return {"accuracy": accuracy}

    def train(self):
        """Fine tune the model with Hugging Face Trainer."""
        if not self.train_texts or self.train_labels is None:
            raise ValueError("Training data is not loaded. Call load_train_dev_test_data() first.")

        if self.model is None or self.tokenizer is None:
            self.load_model()

        train_dataset = self._build_dataset(self.train_texts, self.train_labels)
        eval_dataset = None
        if self.dev_texts and self.dev_labels is not None:
            eval_dataset = self._build_dataset(self.dev_texts, self.dev_labels)

        training_args_kwargs: dict[str, Any] = {
            "output_dir": self.output_dir,
            "per_device_train_batch_size": self.batch_size,
            "per_device_eval_batch_size": self.batch_size,
            "num_train_epochs": self.num_train_epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "seed": self.seed,
            "logging_strategy": "epoch",
            "report_to": [],
        }

        # transformers >= 4.41 renamed evaluation_strategy → eval_strategy
        import inspect
        _ta_params = inspect.signature(TrainingArguments).parameters
        _eval_key = "eval_strategy" if "eval_strategy" in _ta_params else "evaluation_strategy"

        if eval_dataset is not None:
            training_args_kwargs.update(
                {
                    _eval_key: "epoch",
                    "save_strategy": "epoch",
                    "load_best_model_at_end": True,
                    "metric_for_best_model": "accuracy",
                }
            )
        else:
            training_args_kwargs.update(
                {
                    _eval_key: "no",
                    "save_strategy": "no",
                    "load_best_model_at_end": False,
                }
            )

        # Mixed-precision: halves GPU memory usage where supported
        if torch.cuda.is_available():
            training_args_kwargs["fp16"] = True

        # Gradient accumulation: simulate larger effective batches
        # without needing the VRAM to hold them
        if getattr(self, "gradient_accumulation_steps", 1) > 1:
            training_args_kwargs["gradient_accumulation_steps"] = self.gradient_accumulation_steps

        training_args = TrainingArguments(**training_args_kwargs)

        # DataCollatorWithPadding pads each batch to its own longest
        # sequence rather than the global max_length, saving memory.
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics if eval_dataset is not None else None,
            tokenizer=self.tokenizer,
        )

        self.trainer.train()
        return self.trainer

    def predict(self, input_data, predict_batch_size: int | None = None):
        """Predict labels for a string, list of strings, or pandas Series.

        Processes in batches of `predict_batch_size` (defaults to self.batch_size)
        to avoid OOM on machines with limited RAM/VRAM.
        """
        if self.model is None or self.tokenizer is None:
            self.load_model()

        if isinstance(input_data, str):
            texts = [input_data]
        elif isinstance(input_data, pd.Series):
            texts = input_data.fillna("").astype(str).tolist()
        elif isinstance(input_data, (list, tuple)):
            texts = ["" if item is None else str(item) for item in input_data]
        else:
            raise TypeError("input_data must be a string, list/tuple of strings, or pandas Series")

        self.model.eval()
        device = next(self.model.parameters()).device
        bs = predict_batch_size or self.batch_size

        results: list[dict] = []
        for start in range(0, len(texts), bs):
            batch_texts = texts[start : start + bs]
            encoded = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}

            with torch.no_grad():
                logits = self.model(**encoded).logits

            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            pred_ids = probs.argmax(axis=-1)

            for index, pred_id in enumerate(pred_ids):
                pred_id_int = int(pred_id)
                confidence = float(probs[index][pred_id_int])
                label = self.id2label.get(pred_id_int, str(pred_id_int))
                results.append({"label": label, "label_id": pred_id_int, "score": confidence})

        return results if len(results) > 1 else results[0]

    def cleanup(self):
        """Free model, tokenizer, and GPU memory."""
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        self.trainer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()

    def __str__(self):
        return (
            f"HFModel(model_name={self.model_name}, "
            f"text_column={self.text_column}, label_column={self.label_column}, "
            f"num_labels={len(self.label2id) if self.label2id else 'unknown'})"
        )
    
