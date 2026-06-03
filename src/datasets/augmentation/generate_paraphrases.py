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

USER_INSTRUCTIONS = cfg["user_instructions"]


# =========================================================
# Groq Client
# =========================================================

# Shared Groq client instance
client = Groq()


# =========================================================
# Prompt Builder
# =========================================================

def build_prompt(
    question: str,
    regime: str,
) -> str:
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

    instruction = USER_INSTRUCTIONS[regime]

    return (
        f"Query regime: {regime}\n\n"
        f"{instruction}\n\n"
        f"Generate {N_PARAPHRASES} paraphrases.\n\n"
        f'Question:\n"{question}"'
    )


# =========================================================
# LLM Call
# =========================================================

def extract_json_array(content: str) -> List[str]:
    """
    Extract JSON array from model response.
    """

    content = content.strip()

    # Try direct parsing first
    try:

        parsed = json.loads(content)

        if isinstance(parsed, list):
            return parsed

    except json.JSONDecodeError:
        pass

    # Fallback extraction
    start = content.find("[")
    end = content.rfind("]")

    if start == -1 or end == -1:

        raise ValueError(
            f"Model did not return a JSON array.\n\n"
            f"Raw response:\n{content}"
        )

    json_text = content[start:end + 1]

    try:

        parsed = json.loads(json_text)

        if not isinstance(parsed, list):

            raise ValueError(
                "Extracted JSON is not a list."
            )

        return parsed

    except json.JSONDecodeError as e:

        raise ValueError(
            f"Invalid JSON returned.\n\n"
            f"JSON fragment:\n{json_text}\n\n"
            f"Raw response:\n{content}\n\n"
            f"Original error: {e}"
        ) from e

def generate_paraphrases(
    question: str,
    regime: str,
) -> List[str]:
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
                "content": build_prompt(
                    question=question,
                    regime=regime,
                ),
            },
        ],
    )

    content = response.choices[0].message.content.strip()

    return extract_json_array(content)

    content = response.choices[0].message.content.strip()


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
            regime = sample["regime"]

            try:

                # generate semantic paraphrases
                paraphrases = generate_paraphrases(
                    question=question,
                    regime=regime,
                )

            except Exception as e:

                print("\n====================================")
                print("Error generating paraphrases")
                print("====================================")
                print(f"Question: {question}")
                print()
                print(e)
                print()

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
