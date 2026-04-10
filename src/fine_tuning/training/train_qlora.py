"""
QLoRA Fine-Tuning Pipeline for Structured Graph Reasoning
=========================================================

This module implements a QLoRA-based fine-tuning pipeline for training
a causal language model to map natural language questions into a
structured JSON representation of user intent and graph constraints.

The training objective is to teach the model to:

1. read a natural language question
2. infer the user intent
3. generate a strictly valid JSON output containing:
   - user_intent
   - schema

Key Design Principles
---------------------
- memory-efficient training via 4-bit quantization (QLoRA)
- parameter-efficient adaptation using LoRA adapters
- prompt tokens masked during loss computation
- deterministic JSON termination via configurable stop token
- fully configuration-driven experiment setup

Assumptions
-----------
- dataset is pre-cleaned and validated
- train/validation splits are schema-consistent
- no data leakage between splits
- JSON structure is stable across samples

Dataset Format
--------------
Each JSONL entry must contain:

{
    "question": "...",
    "user_intent": "...",
    "schema": {...}
}

Model Output Format
-------------------
The model is trained to generate:

{
    "user_intent": "...",
    "schema": {...}
}
<STOP_TOKEN>

Scope
-----
This module only handles supervised fine-tuning.
All dataset preparation steps occur upstream.
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
    Build the instruction-style prompt provided to the model.

    The prompt follows an instruction-tuning format and separates:
    - task instruction
    - user question
    - model answer region

    Parameters
    ----------
    question : str
        Natural language user question.

    instruction : str
        Task-level instruction describing the expected output structure.

    Returns
    -------
    str
        Formatted prompt string ready for tokenization.
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
    Construct the expected model output string.

    The output is serialized as deterministic JSON followed by a stop token.
    The stop token helps prevent the model from generating trailing text
    after the structured response.

    Parameters
    ----------
    example : Dict
        Dataset row containing user_intent and schema.

    stop_token : str
        Special token indicating the end of the structured output.

    Returns
    -------
    str
        JSON string terminated by the stop token.
    """
    output = {
        "user_intent": example["user_intent"],
        "schema": example["schema"],
    }

    return (
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
            sort_keys=True
        )
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

    # Stop token helps enforce deterministic JSON termination
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
    # Dataset
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
        Tokenize a dataset example and apply label masking.

        The prompt portion is masked using -100 so that loss is computed
        only on the target JSON output.

        This ensures the model learns to generate structured responses
        rather than memorize the instruction text.

        Parameters
        ----------
        example : Dict
            Raw dataset example.

        Returns
        -------
        Dict
            Tokenized inputs including masked labels.
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

        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len
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

    # Disable KV cache during training to avoid incompatibility
    # with gradient checkpointing
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
