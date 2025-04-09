import json
from datetime import datetime
import pandas as pd
from transformers import (
    DataCollatorForSeq2Seq,
    MT5ForConditionalGeneration,
    MT5Tokenizer,
    Trainer,
    TrainingArguments
)
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split


def flatten_output(output: dict) -> str:
    parts = [f"intent: {output['intent']}"]
    for k, v in output.get("entities", {}).items():
        if isinstance(v, list):
            parts.append(f"{k}: {', '.join(v)}")
        else:
            parts.append(f"{k}: {v}")
    return "; ".join(parts)


def preprocess(tokenizer, example):
    model_input = tokenizer(example["input"], max_length=64, truncation=True, padding="max_length")
    labels = tokenizer(
        text_target=example["target"],
        max_length=128,
        truncation=True,
        padding=True
    )
    model_input["labels"] = labels["input_ids"]
    return model_input


class MT5Trainer:
    def __init__(self, model_path: str, log_path: str):
        self.model_name = "google/mt5-small"
        self.model_path = model_path
        self.log_path = log_path

        self.tokenizer = MT5Tokenizer.from_pretrained(self.model_name)
        self.model = MT5ForConditionalGeneration.from_pretrained(self.model_name)

        self.training_args = TrainingArguments(
            output_dir=self.model_path,
            save_strategy="epoch",
            eval_strategy="steps",
            per_device_train_batch_size=8,
            gradient_accumulation_steps=2,
            num_train_epochs=20,
            learning_rate=5e-4,
            warmup_steps=10,
            weight_decay=0.01,
            logging_steps=10,
            save_total_limit=2,
            logging_dir=self.log_path,
            report_to="none",
            max_grad_norm=5.0,
            label_smoothing_factor=0.1
        )

        self.log_history = None
        self.weights = {}

    def train(self, dataset_path: str):
        dataset = self.load_dataset(dataset_path)

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model
        )

        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            data_collator=data_collator,
        )

        self.weights["before"] = self.model.base_model.decoder.embed_tokens.weight[100][:5].tolist()

        trainer.train()

        self.weights["after"] = self.model.base_model.decoder.embed_tokens.weight[100][:5].tolist()

        trainer.save_model(self.model_path)

        self.tokenizer.save_pretrained(self.model_path)
        self.model.save_pretrained(self.model_path)

        self.log_history = trainer.state.log_history
        val_metrics = trainer.evaluate(eval_dataset=dataset["validation"])
        test_metrics = trainer.evaluate(eval_dataset=dataset["test"])

        return self.prep_results(
            val_metrics, test_metrics
        )

    def load_dataset(self, file_path: str, test_split: float = 0.2, validation_split: float = 0.2, seed: int = 42):
        with open(file_path, "r") as f:
            raw_data = json.load(f)

        for ex in raw_data:
            ex["target"] = ex["output"]
            # ex["target"] = flatten_output(ex["output"])
            del ex["output"]

        train_val_data, test_data = train_test_split(raw_data, test_size=test_split, random_state=seed)
        val_relative = validation_split / (1 - test_split)
        train_data, val_data = train_test_split(train_val_data, test_size=val_relative, random_state=seed)

        dataset = DatasetDict({
            "train": Dataset.from_pandas(pd.DataFrame(train_data)),
            "validation": Dataset.from_pandas(pd.DataFrame(val_data)),
            "test": Dataset.from_pandas(pd.DataFrame(test_data))
        })

        tokenized_dataset = dataset.map(lambda x: preprocess(self.tokenizer, x))
        tokenized_dataset["train"] = tokenized_dataset["train"].shuffle(seed=seed)

        return tokenized_dataset

    def prep_results(self, val_metrics, test_metrics):
        log_record = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model_name,
            "num_train_epochs": self.training_args.num_train_epochs,
            "per_device_train_batch_size": self.training_args.per_device_train_batch_size,
            "log_history": self.log_history,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "weights": self.weights
        }
        return log_record

    def load_model(self):
        self.tokenizer = MT5Tokenizer.from_pretrained(self.model_path)
        self.model = MT5ForConditionalGeneration.from_pretrained(self.model_path)

    def generate(self, input_text: str, max_length: int = 128) -> dict:
        self.model.to("cpu")
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True)
        inputs = {k: v.to("cpu") for k, v in inputs.items()}
        output = self.model.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,
            no_repeat_ngram_size=3,
            length_penalty=1,
            repetition_penalty=1.2,
            early_stopping=True
        )

        decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return decoded
