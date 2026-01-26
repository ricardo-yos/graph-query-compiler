"""
Training-Only Question Linguistic Augmentation Pipeline
======================================================

Pipeline responsible for performing *linguistic augmentation exclusively
on the training split* of a question dataset by generating controlled
paraphrases in Brazilian Portuguese.

This module MUST be executed **after** the canonical dataset has been
cleaned, validated, and split into training and validation subsets.
It intentionally operates ONLY on the training data to ensure that:

- Linguistic diversity is increased without semantic drift
- Model robustness to paraphrasing is improved
- The validation set remains pristine and leakage-free

For each training question, this pipeline generates multiple alternative
phrasings while strictly preserving:

- Exact semantic meaning
- Original user intent
- Full compatibility with the same structured schema

Important Characteristics
-------------------------
- Operates exclusively on the training dataset
- Preserves both `user_intent` and `schema` unchanged
- Generates exactly 3 paraphrases per question
- All variations map to the SAME intent and schema
- Enforces strict JSON output validation from the LLM
- Produces natural Brazilian Portuguese (pt-BR)
- Never accesses or modifies validation data

Input
-----
Training dataset (JSONL), where each line contains:
{
    "question": "<natural language question>",
    "user_intent": "<intent type>",
    "schema": <structured intent subset>
}

Output
------
Augmented training dataset (JSONL), containing:
- Original questions
- Three paraphrased variants per question
- Identical `user_intent` and `schema` for all variants
(e.g., `train_augmented.jsonl`)

Notes
-----
- This pipeline increases linguistic surface diversity only
- No new semantic or structural information is introduced
- Designed to prevent paraphrase-based data leakage
"""

import json
from pathlib import Path
from typing import Dict, List

from groq import Groq

from config.env_loader import load_env
from config.paths import SPLITS_DATASETS_DIR, AUGMENTED_DATASETS_DIR


# ==================================================
# Environment & LLM client
# ==================================================

def init_llm_client() -> Groq:
    """
    Initialize the Groq LLM client after loading environment variables.

    Returns
    -------
    Groq
        Initialized Groq client instance.
    """
    load_env()
    return Groq()


# ==================================================
# Prompt construction
# ==================================================

def build_prompt(question: str) -> str:
    """
    Build the prompt used to generate linguistic paraphrases.

    Parameters
    ----------
    question : str
        Original natural language question.

    Returns
    -------
    str
        Prompt instructing the LLM to generate paraphrases.
    """
    return f"""
You are a Brazilian Portuguese language expert.

TASK:
Rewrite the question below into 3 different natural and realistic
variations in Brazilian Portuguese.

RULES:
- Keep EXACTLY the same meaning and intent
- Do NOT add or remove information
- Do NOT change locations, numbers, conditions, or entities
- Do NOT introduce ambiguity
- All variations must be answerable by the SAME schema
- Return ONLY a valid JSON array of 3 strings

QUESTION:
"{question}"
""".strip()


# ==================================================
# LLM interaction
# ==================================================

def call_llm(client: Groq, prompt: str) -> str:
    """
    Call the LLM and return the raw text output.

    Parameters
    ----------
    client : Groq
        Initialized Groq client.
    prompt : str
        Prompt sent to the LLM.

    Returns
    -------
    str
        Raw LLM response content.
    """
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You strictly follow instructions and output only valid JSON."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.7,
        max_tokens=300,
    )

    return response.choices[0].message.content.strip()


def parse_variations(raw_output: str) -> List[str]:
    """
    Parse and validate the LLM output containing question variations.

    Parameters
    ----------
    raw_output : str
        Raw text returned by the LLM.

    Returns
    -------
    List[str]
        List of exactly three paraphrased questions.

    Raises
    ------
    RuntimeError
        If the output is not valid JSON or does not match expectations.
    """
    try:
        variations = json.loads(raw_output)

        if (
            not isinstance(variations, list)
            or len(variations) != 3
            or not all(isinstance(v, str) for v in variations)
        ):
            raise ValueError("Invalid variation format")

        return variations

    except Exception as exc:
        raise RuntimeError(
            f"Failed to parse LLM output:\n{raw_output}"
        ) from exc


def generate_variations(client: Groq, question: str) -> List[str]:
    """
    Generate linguistic paraphrases for a single question.

    Parameters
    ----------
    client : Groq
        Initialized Groq client.
    question : str
        Original question.

    Returns
    -------
    List[str]
        List of paraphrased questions.
    """
    prompt = build_prompt(question)
    raw_output = call_llm(client, prompt)
    return parse_variations(raw_output)


# ==================================================
# Dataset processing
# ==================================================

def process_dataset(
    input_path: Path,
    output_path: Path,
    client: Groq,
) -> None:
    """
    Read the training dataset, generate linguistic paraphrases,
    and write the augmented dataset.

    Parameters
    ----------
    input_path : Path
        Path to the base training dataset (JSONL).
    output_path : Path
        Path where the augmented dataset will be written.
    client : Groq
        Initialized Groq client.
    """
    total = 0
    generated = 0

    with input_path.open("r", encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:

        for line in fin:
            item: Dict = json.loads(line)
            total += 1

            question = item["question"]
            user_intent = item["user_intent"]
            schema = item["schema"]

            # Write original example
            fout.write(
                json.dumps(
                    {
                        "question": question,
                        "user_intent": user_intent,
                        "schema": schema,
                    },
                    ensure_ascii=False
                ) + "\n"
            )

            # Generate and write paraphrased variations
            variations = generate_variations(client, question)

            for variation in variations:
                fout.write(
                    json.dumps(
                        {
                            "question": variation,
                            "user_intent": user_intent,
                            "schema": schema,
                        },
                        ensure_ascii=False
                    ) + "\n"
                )
                generated += 1

    print("=" * 40)
    print("Linguistic Augmentation Summary")
    print("=" * 40)
    print(f"Original questions : {total}")
    print(f"Generated variations: {generated}")
    print(f"Final dataset size : {total + generated}")


# ==================================================
# Entry point
# ==================================================

def main() -> None:
    """
    Execute the training-only linguistic augmentation pipeline.
    """
    input_path = Path(SPLITS_DATASETS_DIR) / "train_base.jsonl"
    output_path = Path(AUGMENTED_DATASETS_DIR) / "train_augmented.jsonl"

    client = init_llm_client()

    process_dataset(
        input_path=input_path,
        output_path=output_path,
        client=client,
    )


if __name__ == "__main__":
    main()
