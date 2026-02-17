"""
QLoRA Adapter Merge Pipeline for Structured Graph Reasoning
===========================================================

This module performs a **permanent merge of LoRA adapters into the base model**,
producing a standalone fully merged model suitable for optimized inference.

Purpose
-------
During training, QLoRA keeps:
- A frozen quantized base model
- Trainable low-rank LoRA adapters

For production deployment, however, it is often preferable to:
- Merge LoRA weights into the base model
- Remove PEFT wrappers
- Save a single consolidated checkpoint

This script performs that merge.

Key Properties
--------------
- Loads base model in full precision (FP16)
- Attaches trained LoRA adapters
- Merges adapter weights into base weights
- Unloads PEFT structure
- Saves a standalone merged model

Why Merge?
----------
Merging provides:
- Lower inference latency
- No PEFT dependency at runtime
- Simpler deployment (e.g., vLLM, llama.cpp conversion)
- Reduced architectural complexity

System Assumptions
------------------
- The adapter directory contains a valid PEFT LoRA checkpoint
- The tokenizer matches the one used during training
- No additional special tokens were introduced post-training

Result
------
A fully merged Hugging Face compatible model directory
ready for inference or further optimization (e.g., GGUF export).
"""

from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from config.paths import MODELS_DIR, LORA_ADAPTER_DIR


# =================================================
# Configuration
# =================================================
# Centralized configuration for reproducibility.
# This keeps merge behavior deterministic and portable.

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
DTYPE = torch.float16
DEVICE_MAP = "auto"

# Output directory for the fully merged model
OUTPUT_DIR = Path(MODELS_DIR) / "graph_compiler_merged"


# =================================================
# Merge Logic
# =================================================

def main() -> None:
    """
    Merge LoRA adapters into the base model and save a standalone checkpoint.

    This procedure:

    1. Loads the original base model.
    2. Loads the tokenizer to ensure vocabulary alignment.
    3. Resizes embedding layers if necessary.
    4. Loads LoRA adapters using PEFT.
    5. Merges adapter weights into base weights.
    6. Unloads the PEFT wrapper.
    7. Saves the merged model and tokenizer.

    Notes
    -----
    - The base model is loaded in FP16 to ensure numerical consistency
      during merging.
    - `merge_and_unload()` permanently integrates LoRA deltas into
      the base model weights.
    - After merging, the resulting model no longer depends on PEFT.
    - The output checkpoint can be used for:
        * Standard Hugging Face inference
        * vLLM serving
        * GGUF conversion (llama.cpp)
        * Further quantization

    Raises
    ------
    RuntimeError
        If adapter loading or merging fails.
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # Load base model
    # -------------------------------------------------
    # We load in FP16 to ensure numerical stability during merge.
    # device_map="auto" allows automatic GPU placement.
    print("\nLoading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=DTYPE,
        device_map=DEVICE_MAP,
        trust_remote_code=True,
    )

    # -------------------------------------------------
    # Load tokenizer
    # -------------------------------------------------
    # The tokenizer must match training configuration exactly.
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        use_fast=True,
    )

    # -------------------------------------------------
    # Embedding alignment check
    # -------------------------------------------------
    # If special tokens were added during fine-tuning,
    # we must resize embeddings before attaching adapters.
    if model.get_input_embeddings().weight.size(0) != len(tokenizer):
        print("Resizing model embeddings to match tokenizer...")
        model.resize_token_embeddings(len(tokenizer))

    # -------------------------------------------------
    # Load LoRA adapters
    # -------------------------------------------------
    # The adapters contain the low-rank deltas learned during fine-tuning.
    print("Loading LoRA adapters...")
    model = PeftModel.from_pretrained(
        model,
        LORA_ADAPTER_DIR,
        ignore_mismatched_sizes=True,
    )

    # -------------------------------------------------
    # Merge adapters
    # -------------------------------------------------
    # This operation:
    #   W_merged = W_base + ΔW_lora
    # After this step, LoRA modules are removed.
    print("Merging adapters into base model...")
    model = model.merge_and_unload()

    # -------------------------------------------------
    # Save merged model
    # -------------------------------------------------
    # safe_serialization=True saves in .safetensors format
    # which is safer and faster to load.
    print("Saving merged model and tokenizer...")
    model.save_pretrained(
        OUTPUT_DIR,
        safe_serialization=True,
    )
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\nMerge completed successfully!")
    print(f"Output model saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
