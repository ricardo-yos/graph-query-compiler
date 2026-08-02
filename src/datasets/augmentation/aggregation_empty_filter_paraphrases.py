"""
Aggregation Empty-Filter Paraphrase Generation
==============================================

LLM-based dataset augmentation module for generating new natural
language examples for aggregation queries without filters.

The generator targets only aggregation query examples where:

    schema.filters == []

Supported query regimes:

- simple_aggregation_query
- relational_aggregation_query

Purpose
-------
Increase dataset coverage for aggregation queries without explicit
filter conditions by generating semantically equivalent paraphrases.

Each generated example preserves exactly:

- query regime
- schema representation
- aggregation semantics

Generation Strategy
-------------------
For each eligible example, multiple paraphrases are generated using
an LLM while keeping the original structured query schema unchanged.

Generated examples are inserted into the final dataset before the
first occurrence of:

    simple_ranking_query

to maintain dataset organization by query regime.

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
- original examples
- generated aggregation examples without filters
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
    BASE_DATASETS_DIR,
    AUGMENTED_DATASETS_DIR,
    DATASETS_CONFIG_DIR,
)


# =========================================================
# Environment
# =========================================================

# Load environment variables (API keys, configuration, etc.)
load_env()


# =========================================================
# Configuration
# =========================================================

CONFIG_PATH = os.path.join(
    DATASETS_CONFIG_DIR,
    "paraphrase_config.yaml"
)


# Load YAML configuration
with open(
    CONFIG_PATH,
    "r",
    encoding="utf-8"
) as f:

    cfg = yaml.safe_load(f)


# Dataset files
INPUT_FILE = Path(
    cfg["input_file"]
)

OUTPUT_FILE = Path(
    cfg["output_file"]
)


# Generated examples are inserted before this regime
INSERT_BEFORE_REGIME = "simple_ranking_query"


# Model configuration
MODEL_NAME = cfg["model_name"]

TEMPERATURE = cfg["temperature"]

MAX_TOKENS = cfg["max_tokens"]


# Number of paraphrases generated per question
N_PARAPHRASES = 15


# Delay between API calls
SLEEP_SECONDS = cfg["sleep_seconds"]


# Prompt configuration
SYSTEM_PROMPT = cfg["system_prompt"]

USER_INSTRUCTIONS = cfg[
    "user_instructions"
]


# Regimes eligible for augmentation
TARGET_REGIMES = {
    "simple_aggregation_query",
    "relational_aggregation_query",
}


# Groq client
client = Groq()



# =========================================================
# Prompt generation
# =========================================================

def build_prompt(
    question: str,
    regime: str,
):
    """
    Build the user prompt sent to the language model.

    Parameters
    ----------
    question:
        Original natural language question.

    regime:
        Query regime associated with the example.

    Returns
    -------
    str
        Formatted prompt containing instructions and question.
    """

    instruction = USER_INSTRUCTIONS[
        regime
    ]


    return (
        f"Query regime: {regime}\n\n"
        f"{instruction}\n\n"
        f"Generate {N_PARAPHRASES} paraphrases.\n\n"
        f'Question:\n"{question}"'
    )



# =========================================================
# JSON extraction
# =========================================================

def extract_json_array(
    content: str
) -> List[str]:
    """
    Extract a JSON list from the model response.

    The function first tries to parse the entire response.
    If additional text exists, it searches for the first
    JSON array block.

    Parameters
    ----------
    content:
        Raw model response.

    Returns
    -------
    list[str]
        Generated paraphrases.
    """

    content = content.strip()


    # Try direct JSON parsing
    try:

        result = json.loads(content)

        if isinstance(result, list):

            return result

    except json.JSONDecodeError:

        pass



    # Fallback: extract JSON array from text
    start = content.find("[")

    end = content.rfind("]")


    if start == -1 or end == -1:

        raise ValueError(
            "Invalid JSON array returned"
        )


    return json.loads(
        content[start:end + 1]
    )



# =========================================================
# Paraphrase generation
# =========================================================

def generate_paraphrases(
    question: str,
    regime: str,
):
    """
    Generate paraphrases for a single question.

    The schema is not modified here. Only the natural
    language question is generated.

    Parameters
    ----------
    question:
        Original question.

    regime:
        Query regime.

    Returns
    -------
    list[str]
        Generated paraphrases.
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
                    question,
                    regime,
                ),
            },

        ],
    )


    return extract_json_array(
        response
        .choices[0]
        .message
        .content
    )



# =========================================================
# Dataset filtering
# =========================================================

def should_process(
    sample
):
    """
    Check whether a dataset sample should receive augmentation.

    Only aggregation queries without filters are selected.

    Parameters
    ----------
    sample:
        Dataset example.

    Returns
    -------
    bool
        True when the sample should be processed.
    """

    if sample["regime"] not in TARGET_REGIMES:

        return False


    filters = sample["schema"].get(
        "filters",
        []
    )


    return filters == []



# =========================================================
# Dataset generation
# =========================================================

def process_dataset():
    """
    Generate paraphrases and create the final augmented dataset.

    Workflow:

    1. Load the original dataset.
    2. Generate paraphrases for eligible aggregation examples.
    3. Insert generated examples before simple_ranking_query.
    4. Preserve all original examples.
    5. Save the final JSONL file.
    """

    input_path = (
        BASE_DATASETS_DIR /
        INPUT_FILE
    )

    output_path = (
        AUGMENTED_DATASETS_DIR /
        OUTPUT_FILE
    )


    # Load original dataset
    with open(
        input_path,
        "r",
        encoding="utf-8"
    ) as fin:

        samples = [
            json.loads(line)
            for line in fin
        ]


    generated_samples = []


    # -----------------------------------------------------
    # Generate paraphrases
    # -----------------------------------------------------

    for sample in tqdm(samples):

        if not should_process(sample):

            continue


        paraphrases = generate_paraphrases(
            question=sample["question"],
            regime=sample["regime"],
        )


        # Create new samples preserving schema
        for text in paraphrases:

            generated_samples.append(
                {
                    "question": text,
                    "regime": sample["regime"],
                    "schema": sample["schema"],
                }
            )


        time.sleep(
            SLEEP_SECONDS
        )



    # -----------------------------------------------------
    # Write final dataset
    # -----------------------------------------------------

    inserted = False


    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as fout:


        for sample in samples:


            # Insert generated block before ranking regime
            if (
                not inserted
                and sample["regime"] == INSERT_BEFORE_REGIME
            ):

                for generated in generated_samples:

                    fout.write(
                        json.dumps(
                            generated,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )


                inserted = True


            # Keep original dataset sample
            fout.write(
                json.dumps(
                    sample,
                    ensure_ascii=False,
                )
                + "\n"
            )



    # -----------------------------------------------------
    # Fallback
    # -----------------------------------------------------

    # If simple_ranking_query does not exist,
    # append generated examples at the end.
    if not inserted:

        with open(
            output_path,
            "a",
            encoding="utf-8"
        ) as fout:

            for generated in generated_samples:

                fout.write(
                    json.dumps(
                        generated,
                        ensure_ascii=False,
                    )
                    + "\n"
                )


    print(
        f"Generated samples: {len(generated_samples)}"
    )

    print(
        f"Saved: {output_path}"
    )



# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    process_dataset()
