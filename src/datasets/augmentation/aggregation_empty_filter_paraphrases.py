"""
Aggregation Empty Filter Paraphrase Generation
==============================================

LLM-based dataset augmentation module for generating new natural
language examples for aggregation queries without filters.

This module expands aggregation query coverage by generating
semantically equivalent paraphrases while preserving the original
structured query schema.

Target Query Regimes
--------------------
The generator processes only:

- simple_aggregation_query
- relational_aggregation_query

with:

    schema.filters == []

Purpose
-------
Increase dataset coverage for aggregation queries without explicit
filter conditions.

Each generated example preserves exactly:

- query regime
- schema representation
- aggregation semantics

Generation Strategy
-------------------
For each eligible example, multiple paraphrases are generated using
an LLM. The generated questions receive the same structural schema
as the original example.

The generated aggregation examples are stored separately and later
merged into the complete question dataset before:

    simple_ranking_query

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
Generated JSONL dataset containing:
- original aggregation examples
- generated aggregation paraphrases
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

# Load environment variables required by the application
# (e.g., API keys and external service configuration)
load_env()



# =========================================================
# Configuration
# =========================================================

# Path to YAML configuration file containing:
# - model configuration
# - prompt configuration
# - dataset paths
CONFIG_PATH = os.path.join(
    DATASETS_CONFIG_DIR,
    "paraphrase_config.yaml"
)


# Load pipeline configuration
with open(
    CONFIG_PATH,
    "r",
    encoding="utf-8"
) as f:

    cfg = yaml.safe_load(f)



# Input dataset containing structured query examples
INPUT_FILE = Path(
    cfg["input_file"]
)


# Output dataset containing aggregation paraphrases
OUTPUT_FILE = Path(
    "aggregation_empty_paraphrases.jsonl"
)

FINAL_OUTPUT_FILE = Path(
    "questions_paraphrased.jsonl"
)

INSERT_BEFORE_REGIME = "simple_ranking_query"


# LLM configuration
MODEL_NAME = cfg["model_name"]

TEMPERATURE = cfg["temperature"]

MAX_TOKENS = cfg["max_tokens"]


# Number of paraphrases generated per original question
N_PARAPHRASES = 15


# Delay between API calls to control request rate
SLEEP_SECONDS = cfg["sleep_seconds"]



# Prompt configuration
SYSTEM_PROMPT = cfg["system_prompt"]

USER_INSTRUCTIONS = cfg[
    "user_instructions"
]



# Only aggregation regimes without filters are augmented
TARGET_REGIMES = {
    "simple_aggregation_query",
    "relational_aggregation_query",
}



# Initialize Groq API client
client = Groq()



# =========================================================
# Prompt
# =========================================================

def build_prompt(
    question: str,
    regime: str,
):
    """
    Build the prompt sent to the LLM.

    The prompt includes:
    - query regime
    - regime-specific instructions
    - original question

    The generated paraphrases must preserve the original
    query meaning.
    """

    # Retrieve instructions specific to the current regime
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
    Extract the generated paraphrases from the LLM response.

    Handles responses where:
    - the model returns a valid JSON array directly
    - the JSON array is surrounded by additional text

    Returns:
        List of generated paraphrase strings.
    """

    content = content.strip()


    # Try parsing the complete response first
    try:

        result = json.loads(content)

        if isinstance(result, list):

            return result

    except json.JSONDecodeError:

        pass



    # Fallback: locate JSON array boundaries
    start = content.find("[")

    end = content.rfind("]")


    if start == -1 or end == -1:

        raise ValueError(
            "Invalid JSON array returned"
        )


    return json.loads(
        content[start:end+1]
    )



# =========================================================
# Generate paraphrases
# =========================================================

def generate_paraphrases(
    question: str,
    regime: str,
):
    """
    Generate semantic paraphrases for a single aggregation query.

    Only the natural language question is generated.
    The original structured schema is preserved during
    dataset creation.

    Returns:
        List of paraphrased questions.
    """

    response = client.chat.completions.create(

        # LLM model used for paraphrase generation
        model=MODEL_NAME,

        # Controls randomness of generated paraphrases
        temperature=TEMPERATURE,

        # Maximum response size
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


    # Convert model response into a Python list
    return extract_json_array(
        response
        .choices[0]
        .message
        .content
    )



# =========================================================
# Filter
# =========================================================

def should_process(
    sample
):
    """
    Determine whether a dataset example should be augmented.

    An example is selected only when:

    - its regime is an aggregation regime
    - schema.filters is empty

    This module specifically targets aggregation examples
    without explicit filter constraints.
    """


    # Ignore non-aggregation regimes
    if sample["regime"] not in TARGET_REGIMES:

        return False


    # Retrieve filters from schema
    filters = sample["schema"].get(
        "filters",
        []
    )


    # Only process examples without filters
    return filters == []



# =========================================================
# Dataset processing
# =========================================================

def process_dataset():
    """
    Generate aggregation paraphrases and save augmentation data.

    Workflow:

    1. Read the structured query dataset.
    2. Select aggregation examples without filters.
    3. Generate paraphrases using the LLM.
    4. Preserve the original schema for generated examples.
    5. Save original and generated examples as JSONL.
    """


    # Build complete input and output paths
    input_path = (
        BASE_DATASETS_DIR /
        INPUT_FILE
    )


    output_path = (
        AUGMENTED_DATASETS_DIR /
        OUTPUT_FILE
    )


    # Counter for generated dataset size
    total = 0



    # Read input dataset and write augmented dataset
    with open(
        input_path,
        "r",
        encoding="utf-8"
    ) as fin, open(
        output_path,
        "w",
        encoding="utf-8"
    ) as fout:


        # Process each JSONL example
        for line in tqdm(fin):


            sample = json.loads(line)


            # Skip examples outside augmentation criteria
            if not should_process(sample):

                continue



            # Generate semantic variations
            paraphrases = generate_paraphrases(

                question=sample["question"],

                regime=sample["regime"],

            )



            # Save original example
            # The schema remains unchanged
            fout.write(
                json.dumps(
                    sample,
                    ensure_ascii=False,
                )
                + "\n"
            )

            total += 1



            # Save generated questions while preserving the original schema
            for text in paraphrases:


                # Create new example preserving
                # regime and schema
                new_sample = {

                    "question": text,

                    "regime": sample["regime"],

                    "schema": sample["schema"],

                }


                fout.write(

                    json.dumps(
                        new_sample,
                        ensure_ascii=False,
                    )
                    + "\n"

                )


                total += 1



            # Avoid exceeding API rate limits
            time.sleep(
                SLEEP_SECONDS
            )



    print(
        f"Generated samples: {total}"
    )


def merge_paraphrases_before_regime():
    """
    Merge generated aggregation paraphrases into the final dataset.

    The generated examples are inserted before the first occurrence
    of simple_ranking_query to preserve the organization of query
    regimes in the dataset.

    Output:
        questions_paraphrased.jsonl
    """

    base_path = (
        BASE_DATASETS_DIR /
        INPUT_FILE
    )

    paraphrase_path = (
        AUGMENTED_DATASETS_DIR /
        OUTPUT_FILE
    )

    final_path = (
        AUGMENTED_DATASETS_DIR /
        FINAL_OUTPUT_FILE
    )


    # Load augmentation examples generated from aggregation queries
    # without filters
    with open(
        paraphrase_path,
        "r",
        encoding="utf-8"
    ) as f:

        generated_samples = [
            json.loads(line)
            for line in f
        ]


    inserted = False


    # Create final dataset
    with open(
        base_path,
        "r",
        encoding="utf-8"
    ) as fin, open(
        final_path,
        "w",
        encoding="utf-8"
    ) as fout:


        for line in fin:

            sample = json.loads(line)


            # Insert aggregation augmentation examples before ranking regime
            # to preserve dataset regime ordering
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


            # Write original example
            fout.write(
                json.dumps(
                    sample,
                    ensure_ascii=False,
                )
                + "\n"
            )


    # Fallback if ranking regime was not found
    if not inserted:

        with open(
            final_path,
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
        f"Saved merged dataset: {final_path}"
    )


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    process_dataset()

    merge_paraphrases_before_regime()
