"""
QLoRA Inference Pipeline for Structured Graph Reasoning
=======================================================

This module implements a **QLoRA-based inference pipeline** for converting
natural language questions into structured JSON outputs suitable for
graph query execution and reasoning.

The model is trained to:
- Interpret a natural language question
- Infer the user's intent
- Produce a strictly valid JSON output containing:
  - `user_intent`
  - `schema`

Key Design Principles
---------------------
- Uses **4-bit quantized base model (QLoRA)** for memory efficiency
- Loads **LoRA adapters fine-tuned for intent and schema prediction**
- Enforces deterministic JSON termination via a configurable stop token
- Fully configuration-driven via YAML
- Optimized for **low-latency, deterministic inference**

System Guarantees
-----------------
- Deterministic decoding (no sampling)
- Strict JSON output enforcement
- Robust post-processing and parsing
- Compatibility with downstream symbolic pipelines

Assumptions
-----------
- The model was trained using the same stop token.
- Tokenizer vocabulary matches training configuration.
- The adapter directory contains a valid PEFT checkpoint.
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
# Centralized configuration loading for inference.
# This enables:
# - Deterministic reproducibility
# - Easy model swapping
# - Rapid experimentation
# - Safe production deployment via config isolation

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
# Load tokenizer from adapter directory to ensure
# full vocabulary alignment with fine-tuning.

tokenizer = AutoTokenizer.from_pretrained(
    ADAPTER_DIR,
    use_fast=True,
)
tokenizer.pad_token = tokenizer.eos_token

# Ensure tokenizer vocabulary contains the stop token.
# This is critical for deterministic generation termination.
if STOP_TOKEN not in tokenizer.get_vocab():
    tokenizer.add_tokens([STOP_TOKEN], special_tokens=False)

stop_token_id = tokenizer.convert_tokens_to_ids(STOP_TOKEN)


# =================================================
# Quantization (QLoRA)
# =================================================
# Configure 4-bit quantization using bitsandbytes.
# This drastically reduces memory usage while preserving
# high-quality inference performance.

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
# Load the quantized base model and align embedding
# matrices to match tokenizer vocabulary.

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
# Attach fine-tuned LoRA adapters responsible for
# structured intent and schema generation.

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
    Build a fully formatted instruction prompt for inference.

    This function ensures that the model receives:
    - Clear task definition
    - Consistent prompt structure
    - Deterministic formatting identical to training

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

def extract_first_json(text: str) -> str:
    """
    Extract the first complete JSON object found in a text sequence.

    This method is robust against:
    - Leading text
    - Trailing explanations
    - Hallucinated completions
    - Missing stop tokens

    It guarantees that only the first syntactically complete JSON
    object is returned.

    Parameters
    ----------
    text : str
        Raw decoded model output.

    Returns
    -------
    str
        Extracted JSON string.

    Raises
    ------
    ValueError
        If no complete JSON object is found.
    """
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in output")

    stack = []
    for i in range(start, len(text)):
        if text[i] == "{":
            stack.append("{")
        elif text[i] == "}":
            stack.pop()
            if not stack:
                return text[start:i + 1]

    raise ValueError("Unclosed JSON object in output")


@torch.no_grad()
def predict(
    question: str,
    debug: bool = False,
) -> Union[Dict, str]:
    """
    Run deterministic inference and return structured JSON output.

    The inference process:
    - Builds a deterministic instruction prompt
    - Performs greedy decoding (no sampling)
    - Enforces stop-token termination
    - Applies robust JSON extraction and parsing

    Parameters
    ----------
    question : str
        Natural language input question.
    debug : bool, optional
        If True, returns raw model output upon parsing failure.

    Returns
    -------
    Dict
        Dictionary containing `user_intent` and `schema`.

    Raises
    ------
    json.JSONDecodeError
        If JSON parsing fails and debug=False.
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
        json_text = extract_first_json(text)
        parsed = json.loads(json_text)

        return {
            "user_intent": parsed.get("user_intent"),
            "schema": parsed.get("schema"),
        }

    except Exception as e:
        if debug:
            return {
                "error": "Invalid JSON generated",
                "raw_output": text,
                "exception": str(e),
            }
        raise


# =================================================
# CLI
# =================================================

def main() -> None:
    """
    Interactive command-line interface for real-time inference.

    This interface allows:
    - Manual testing
    - Prompt debugging
    - Model behavior inspection
    """
    while True:
        question = input("\nPergunta (ENTER para sair): ").strip()
        if not question:
            break

        result = predict(question, debug=True)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
