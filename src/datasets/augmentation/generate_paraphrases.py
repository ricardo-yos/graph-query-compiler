"""
Paraphrase Generation Pipeline
==============================

LLM-based dataset augmentation pipeline for generating natural
language paraphrases from structured graph query datasets.

This module expands an existing question dataset by generating
multiple semantically equivalent paraphrases for each question
while preserving the original structured schema.

Purpose
-------
Increase linguistic diversity for downstream NLP tasks such as:

- semantic parsing
- natural language understanding
- instruction tuning
- graph query generation
- query generalization

Each generated paraphrase preserves:
- the original query semantics
- the structural regime
- the associated schema representation

Pipeline Characteristics
------------------------
- deterministic dataset structure preservation
- semantic augmentation only
- schema-consistent paraphrase generation
- configurable paraphrase count
- Groq API integration
- JSONL-based streaming processing

Input
-----
JSONL dataset containing:

{
    "question": "...",
    "regime": "...",
    "schema": {...}
}

Output
------
Expanded JSONL dataset containing:
- original questions
- generated paraphrases
- preserved schema metadata

Dependencies
------------
- groq
- yaml
- tqdm
"""

import json
import os
import time
import yaml

from pathlib import Path
from typing import List

from tqdm import tqdm
from groq import Groq

from config.env_loader import load_env
from config.paths import (
    AUGMENTED_DATASETS_DIR,
    BASE_DATASETS_DIR,
    DATASETS_CONFIG_DIR,
)


# =========================================================
# Environment
# =========================================================

# Load environment variables before creating API clients
load_env()


# =========================================================
# Configuration
# =========================================================

# Centralized configuration for reproducibility
CONFIG_PATH = os.path.join(
    DATASETS_CONFIG_DIR,
    "paraphrase_config.yaml"
)

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# input/output dataset paths
INPUT_FILE = Path(cfg["input_file"])

OUTPUT_FILE = Path(cfg["output_file"])

# LLM generation parameters
MODEL_NAME = cfg["model_name"]

TEMPERATURE = cfg["temperature"]

MAX_TOKENS = cfg["max_tokens"]

N_PARAPHRASES = cfg["n_paraphrases"]

SLEEP_SECONDS = cfg["sleep_seconds"]

# prompting configuration
SYSTEM_PROMPT = cfg["system_prompt"]

USER_INSTRUCTION = cfg["user_instruction"]


# =========================================================
# Groq Client
# =========================================================

# Shared Groq client instance
client = Groq()


# =========================================================
# Prompt Builder
# =========================================================

def build_prompt(question: str) -> str:
    """
    Build deterministic paraphrasing prompt.

    The prompt instructs the LLM to generate multiple
    semantically equivalent paraphrases while preserving
    the original query meaning.

    Parameters
    ----------
    question : str
        Original natural language question.

    Returns
    -------
    str
        Formatted LLM prompt.
    """

    return f"""
{USER_INSTRUCTION}

Generate {N_PARAPHRASES} paraphrases.

Question:
"{question}"
"""


# =========================================================
# LLM Call
# =========================================================

def generate_paraphrases(question: str) -> List[str]:
    """
    Generate paraphrases using the Groq API.

    The model is expected to return a JSON array
    containing paraphrased questions.

    Parameters
    ----------
    question : str
        Original question.

    Returns
    -------
    List[str]
        Generated paraphrases.

    Raises
    ------
    ValueError
        If the model response does not contain
        a valid JSON array.
    """

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": build_prompt(question),
            },
        ],
    )

    content = response.choices[0].message.content.strip()

    # -----------------------------------------------------
    # Robust JSON extraction
    # -----------------------------------------------------
    # Some LLM responses may include explanations or
    # formatting outside the JSON array.
    # Extract only the array portion safely.
    start = content.find("[")

    end = content.rfind("]")

    if start == -1 or end == -1:
        raise ValueError(
            "Model did not return valid JSON array."
        )

    json_text = content[start:end + 1]

    return json.loads(json_text)


# =========================================================
# Dataset Processing
# =========================================================

def process_dataset() -> None:
    """
    Expand a structured question dataset with paraphrases.

    Workflow
    --------
    1. Load base dataset
    2. Generate paraphrases for each question
    3. Preserve original schema metadata
    4. Save original + paraphrased samples
    5. Persist expanded dataset as JSONL

    The resulting dataset maintains structural alignment
    between:
    - question
    - regime
    - schema

    Returns
    -------
    None
    """

    input_path = os.path.join(
        BASE_DATASETS_DIR,
        INPUT_FILE,
    )

    output_path = os.path.join(
        AUGMENTED_DATASETS_DIR,
        OUTPUT_FILE,
    )

    total_written = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        lines = fin.readlines()

        # -------------------------------------------------
        # Process dataset sequentially
        # -------------------------------------------------

        for line in tqdm(lines):

            sample = json.loads(line)

            question = sample["question"]

            try:

                # generate semantic paraphrases
                paraphrases = generate_paraphrases(
                    question
                )

            except Exception as e:

                # skip malformed or failed generations
                print("\nError generating paraphrases")
                print(question)
                print(e)

                continue

            # =================================================
            # Save original sample
            # =================================================

            fout.write(
                json.dumps(
                    sample,
                    ensure_ascii=False,
                ) + "\n"
            )

            total_written += 1

            # =================================================
            # Save paraphrased samples
            # =================================================

            for paraphrase in paraphrases:

                new_sample = {
                    "question": paraphrase,
                    "regime": sample["regime"],
                    "schema": sample["schema"],
                }

                fout.write(
                    json.dumps(
                        new_sample,
                        ensure_ascii=False,
                    ) + "\n"
                )

                total_written += 1

            # small delay to avoid API burst spikes
            time.sleep(SLEEP_SECONDS)

    print("\nDone.")
    print(f"Total samples written: {total_written}")


# =========================================================
# Entry Point
# =========================================================

if __name__ == "__main__":

    process_dataset()
