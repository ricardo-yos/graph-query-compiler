"""
QLoRA Fine-Tuning Pipeline for Structured Graph Reasoning
=========================================================

This module implements a **QLoRA-based fine-tuning pipeline** for training
a causal language model to map **natural language questions** into a
**structured JSON schema** representing user intent and graph constraints.

The training objective is to condition the model to:
- Read a natural language question
- Infer the user's intent
- Produce a strictly valid JSON output containing:
  - `user_intent`
  - `schema`

Key Design Principles
---------------------
- Uses **4-bit quantization (QLoRA)** for memory-efficient fine-tuning
- Applies **LoRA adapters** to selected transformer modules
- Masks the prompt portion of the sequence during loss computation
- Enforces deterministic JSON termination using a configurable stop token
- Fully configuration-driven via YAML

This pipeline assumes that:
- The dataset has already been cleaned and validated
- Train and validation splits are schema-consistent
- No data leakage exists between splits

Dataset Format
--------------
Each JSONL example must follow the structure:

{
    "question": "<natural language question>",
    "user_intent": "<intent label>",
    "schema": <structured intent subset>
}

Output Format
-------------
The model is trained to generate:

{
    "user_intent": "...",
    "schema": {...}
}
<STOP_TOKEN>

Notes
-----
- This script focuses exclusively on supervised fine-tuning
- Linguistic augmentation and dataset generation occur upstream
- Evaluation is performed during training using a validation split
"""

import json
import yaml
import torch
from pathlib import Path
from typing import Dict

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from config.paths import FINE_TUNING_CONFIG_DIR


# ==================================================
# Configuration loader
# ==================================================

def load_config(path: Path) -> Dict:
    """
    Load and normalize the YAML configuration file.

    Parameters
    ----------
    path : Path
        Path to the YAML configuration file.

    Returns
    -------
    Dict
        Parsed and type-normalized configuration dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Float parameters
    for key in ["learning_rate"]:
        if key in cfg:
            cfg[key] = float(cfg[key])

    # Integer parameters
    for key in [
        "batch_size",
        "gradient_accumulation_steps",
        "num_epochs",
        "sequence_len",
        "logging_steps",
        "save_steps",
        "save_total_limit",
        "eval_steps",
        "max_steps",
        "warmup_steps",
    ]:
        if key in cfg:
            cfg[key] = int(cfg[key])

    # Boolean parameters
    for key in ["fp16", "bf16", "load_in_4bit", "bnb_4bit_use_double_quant"]:
        if key in cfg:
            cfg[key] = bool(cfg[key])

    return cfg


# ==================================================
# Prompt & output formatting
# ==================================================

def build_prompt(question: str, instruction: str) -> str:
    """
    Construct the full instruction prompt for the model.

    Parameters
    ----------
    question : str
        Natural language user question.
    instruction : str
        Task-level instruction describing the expected behavior.

    Returns
    -------
    str
        Fully formatted prompt string.
    """
    return (
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Question:\n"
        f"{question}\n\n"
        "### Answer:\n"
    )


def build_output(example: Dict, stop_token: str) -> str:
    """
    Build the expected model output string.

    Parameters
    ----------
    example : Dict
        Dataset example containing `user_intent` and `schema`.
    stop_token : str
        Token that deterministically terminates the JSON output.

    Returns
    -------
    str
        JSON-formatted output followed by the stop token.
    """
    output = {
        "user_intent": example["user_intent"],
        "schema": example["schema"],
    }

    return (
        json.dumps(output, ensure_ascii=False, separators=(",", ":"))
        + "\n"
        + stop_token
    )


# ==================================================
# Training pipeline
# ==================================================

def main() -> None:
    """
    Execute the QLoRA fine-tuning pipeline.
    """
    cfg = load_config(
        Path(FINE_TUNING_CONFIG_DIR) / "training" / "qlora_config.yaml"
        )

    stop_token = cfg["stop_token"]

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # Tokenizer
    # -------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["base_model"],
        use_fast=True,
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens([stop_token], special_tokens=False)

    # -------------------------------------------------
    # Quantization (QLoRA)
    # -------------------------------------------------
    compute_dtype = torch.float16
    if cfg.get("bnb_4bit_compute_dtype") == "bfloat16":
        compute_dtype = torch.bfloat16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg["load_in_4bit"],
        bnb_4bit_quant_type=cfg["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=cfg["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=compute_dtype,
    )

    # -------------------------------------------------
    # Base model
    # -------------------------------------------------
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
    )

    model.resize_token_embeddings(len(tokenizer))
    model = prepare_model_for_kbit_training(model)

    # -------------------------------------------------
    # LoRA adapters
    # -------------------------------------------------
    lora_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"],
        bias=cfg.get("lora_bias", "none"),
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # -------------------------------------------------
    # Dataset loading
    # -------------------------------------------------
    dataset = load_dataset(
        "json",
        data_files={
            "train": cfg["dataset"]["data_files"]["train"],
            "validation": cfg["dataset"]["data_files"]["validation"],
        },
    )

    instruction = cfg["task_instruction"]
    max_length = cfg["sequence_len"]

    def tokenize_fn(example: Dict) -> Dict:
        """
        Tokenize a dataset example and mask the prompt portion.

        Parameters
        ----------
        example : Dict
            Raw dataset example.

        Returns
        -------
        Dict
            Tokenized example with labels.
        """
        prompt = build_prompt(example["question"], instruction)
        output = build_output(example, stop_token)

        full_text = prompt + output

        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

        labels = tokenized["input_ids"].copy()

        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=False,
        )["input_ids"]

        labels[: len(prompt_ids)] = [-100] * len(prompt_ids)
        tokenized["labels"] = labels

        return tokenized

    tokenized_dataset = dataset.map(
        tokenize_fn,
        remove_columns=dataset["train"].column_names,
        batched=False,
    )

    # -------------------------------------------------
    # Training arguments
    # -------------------------------------------------
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg.get(
            "eval_batch_size", cfg["batch_size"]
        ),
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["num_epochs"],
        max_steps=cfg.get("max_steps", -1),
        lr_scheduler_type=cfg.get("lr_scheduler", "constant"),
        warmup_steps=cfg.get("warmup_steps", 0),
        fp16=cfg.get("fp16", False),
        bf16=cfg.get("bf16", False),
        logging_steps=cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=cfg["eval_steps"],
        save_strategy="steps",
        save_steps=cfg["save_steps"],
        save_total_limit=cfg["save_total_limit"],
        optim=cfg.get("optim", "paged_adamw_8bit"),
        load_best_model_at_end=True,
        report_to="none",
    )

    # -------------------------------------------------
    # Trainer
    # -------------------------------------------------
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
    )

    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Training finished. Model saved to {output_dir}")


if __name__ == "__main__":
    main()
