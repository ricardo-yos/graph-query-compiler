"""
QLoRA Adapter Merge Pipeline
============================

Utility script responsible for merging a trained QLoRA adapter
into the original base language model.

Workflow
--------
1. Load tokenizer saved during fine-tuning
2. Load original base model
3. Validate tokenizer/model vocabulary compatibility
4. Resize embeddings if new tokens were added
5. Load LoRA adapter weights
6. Merge adapter weights into the base model
7. Save the standalone merged model
8. Save the tokenizer

Output
------
Standalone HuggingFace model available at:

models/gqc-merged/

The resulting model no longer requires PEFT adapters
for inference.
"""

from pathlib import Path

import torch

from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from config.paths import (
    MODELS_DIR,
    LORA_ADAPTER_DIR,
)


# ============================================================
# Model configuration
# ============================================================

# Original foundation model used during QLoRA fine-tuning.
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

# Output directory containing the standalone merged model.
OUTPUT_DIR = Path(MODELS_DIR) / "gqc-merged"


# ============================================================
# Generation configuration
# ============================================================

# Custom stop token added during fine-tuning.
# The token must exist in the tokenizer before merging.
STOP_TOKEN = "</json>"


# ============================================================
# Merge pipeline
# ============================================================

def main() -> None:
    """
    Merge QLoRA adapters into the base model.

    This function creates a standalone model by combining
    the original foundation model weights with the learned
    LoRA adapter parameters.

    The resulting model can be loaded directly for inference
    without requiring the PEFT adapter separately.
    """

    # --------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        LORA_ADAPTER_DIR,
        use_fast=True,
    )


    # --------------------------------------------------------
    # Validate custom tokens
    # --------------------------------------------------------

    token_id = tokenizer.convert_tokens_to_ids(
        STOP_TOKEN
    )

    if token_id == tokenizer.unk_token_id:
        raise ValueError(
            f"Token '{STOP_TOKEN}' not found in tokenizer."
        )

    print(
        f"Stop token '{STOP_TOKEN}' found with id {token_id}"
    )


    # --------------------------------------------------------
    # Load base model
    # --------------------------------------------------------

    print("Loading base model...")

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map=None,
        trust_remote_code=True,
    )


    # --------------------------------------------------------
    # Check tokenizer/model vocabulary compatibility
    # --------------------------------------------------------

    embedding_size = (
        base_model.get_input_embeddings()
        .weight.size(0)
    )

    tokenizer_size = len(tokenizer)

    print(f"Model embeddings: {embedding_size}")
    print(f"Tokenizer size: {tokenizer_size}")


    # Resize embeddings when new special tokens were added
    # during fine-tuning.
    if embedding_size != tokenizer_size:

        print(
            "Resizing model embeddings "
            "to match tokenizer..."
        )

        base_model.resize_token_embeddings(
            tokenizer_size,
            mean_resizing=False,
        )


    # --------------------------------------------------------
    # Load LoRA adapter
    # --------------------------------------------------------

    print("Loading QLoRA adapters...")

    model = PeftModel.from_pretrained(
        base_model,
        LORA_ADAPTER_DIR,
    )


    # --------------------------------------------------------
    # Merge adapter weights
    # --------------------------------------------------------

    print("Merging adapters into base model...")

    merged_model = model.merge_and_unload()


    # --------------------------------------------------------
    # Save final model
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Saving merged model...")

    merged_model.save_pretrained(
        OUTPUT_DIR,
        safe_serialization=True,
    )


    # Save tokenizer together with the model because
    # vocabulary must match the embedding matrix.
    print("Saving tokenizer...")

    tokenizer.save_pretrained(
        OUTPUT_DIR
    )


    print(
        f"Model successfully saved to {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
