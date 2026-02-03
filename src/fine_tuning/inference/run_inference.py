"""
QLoRA Inference Pipeline for Structured Graph Reasoning
=======================================================

This module implements a **QLoRA-based inference pipeline** for generating
structured JSON outputs from natural language questions.

The model is trained to:
- Read a natural language question
- Infer the user's intent
- Produce a strictly valid JSON output containing:
  - `user_intent`
  - `schema`

Key Design Principles
---------------------
- Uses **4-bit quantized base model (QLoRA)**
- Loads **LoRA adapters trained for intent + schema prediction**
- Enforces deterministic JSON termination using a configurable stop token
- Fully configuration-driven via YAML
- Designed for **low-latency, deterministic inference**

This script assumes that:
- The model was trained using the same stop token
- Tokenizer vocabulary is aligned with training
- The adapter directory contains a valid PEFT checkpoint
"""

import json
import yaml
import torch
from pathlib import Path
from typing import Dict, Union

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

from config.paths import FINE_TUNING_CONFIG_DIR


# =================================================
# Configuration
# =================================================

CONFIG_PATH = Path(FINE_TUNING_CONFIG_DIR) / "inference" / "inference_config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg: Dict = yaml.safe_load(f)

BASE_MODEL = cfg["base_model"]
ADAPTER_DIR = Path(cfg["adapter_dir"])
MAX_NEW_TOKENS = cfg["max_new_tokens"]

STOP_TOKEN = cfg["stop_token"]
ANSWER_MARKER = cfg["answer_marker"]
TASK_INSTRUCTION = cfg["task_instruction"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =================================================
# Tokenizer
# =================================================

tokenizer = AutoTokenizer.from_pretrained(
    ADAPTER_DIR,
    use_fast=True,
)
tokenizer.pad_token = tokenizer.eos_token

# Ensure tokenizer vocabulary is aligned with training
if STOP_TOKEN not in tokenizer.get_vocab():
    tokenizer.add_tokens([STOP_TOKEN], special_tokens=False)

stop_token_id = tokenizer.convert_tokens_to_ids(STOP_TOKEN)


# =================================================
# Quantization (QLoRA)
# =================================================

qcfg = cfg["quantization"]

compute_dtype = torch.float16
if qcfg["bnb_4bit_compute_dtype"] == "bfloat16":
    compute_dtype = torch.bfloat16

bnb_config = BitsAndBytesConfig(
    load_in_4bit=qcfg["load_in_4bit"],
    bnb_4bit_quant_type=qcfg["bnb_4bit_quant_type"],
    bnb_4bit_use_double_quant=qcfg["bnb_4bit_use_double_quant"],
    bnb_4bit_compute_dtype=compute_dtype,
)


# =================================================
# Base model loading
# =================================================

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)

# Critical: align embedding matrix with tokenizer
base_model.resize_token_embeddings(len(tokenizer))


# =================================================
# Load LoRA adapters
# =================================================

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_DIR,
)

model.eval()
model.config.use_cache = True


# =================================================
# Prompt formatting
# =================================================

def build_prompt(question: str) -> str:
    """
    Build the instruction prompt for inference.

    Parameters
    ----------
    question : str
        Natural language user question.

    Returns
    -------
    str
        Fully formatted instruction prompt.
    """
    return (
        "### Instruction:\n"
        f"{TASK_INSTRUCTION}\n\n"
        "### Question:\n"
        f"{question}\n\n"
        "### Answer:\n"
    )


# =================================================
# Inference
# =================================================

@torch.no_grad()
def predict(
    question: str,
    debug: bool = False,
) -> Union[Dict, str]:
    """
    Run deterministic inference and return structured JSON output.

    Parameters
    ----------
    question : str
        Natural language question.
    debug : bool, optional
        Whether to return raw output if JSON parsing fails.

    Returns
    -------
    Dict
        Dictionary containing `user_intent` and `schema`.

    Raises
    ------
    json.JSONDecodeError
        If the model output cannot be parsed and debug=False.
    """
    prompt = build_prompt(question)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(DEVICE)

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        eos_token_id=stop_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    decoded = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )

    # ---------------------------------------------
    # Extract model answer
    # ---------------------------------------------
    if ANSWER_MARKER in decoded:
        text = decoded.split(ANSWER_MARKER, 1)[-1]
    else:
        text = decoded

    if STOP_TOKEN in text:
        text = text.split(STOP_TOKEN, 1)[0]

    text = text.strip()

    # ---------------------------------------------
    # JSON parsing
    # ---------------------------------------------
    try:
        parsed = json.loads(text)
        return {
            "user_intent": parsed.get("user_intent"),
            "schema": parsed.get("schema"),
        }

    except json.JSONDecodeError:
        if debug:
            return {
                "error": "Invalid JSON generated",
                "raw_output": text,
            }
        raise


# =================================================
# CLI
# =================================================

def main() -> None:
    """
    Interactive command-line interface for inference.
    """
    while True:
        question = input("\nPergunta (ENTER para sair): ").strip()
        if not question:
            break

        result = predict(question, debug=True)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
