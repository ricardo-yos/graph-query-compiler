"""
Question Dataset Generation Pipeline
====================================

Deterministic pipeline for generating natural language questions
(Brazilian Portuguese) from structured graph query intents.

This module transforms validated and quality-filtered intents into
natural, human-like questions that a user would realistically ask
when interacting with a graph-backed system.

Each generated example contains:
- A single natural language question (pt-BR)
- The corresponding structured intent schema used to generate it

This dataset is designed for:
- LLM fine-tuning
- Instruction-following alignment
- Semantic parsing and intent understanding tasks

Important Characteristics
-------------------------
- Generates ONLY questions (no answers, no Cypher, no explanations)
- Preserves the original intent schema for full traceability
- Uses deterministic, rule-based prompting for consistency
- Avoids hallucination by strictly using intent-provided information
- Optimized for graph query understanding in real-world scenarios

Input
-----
- JSONL file containing semantically validated and quality-filtered intents

Output
------
- JSONL dataset where each line contains:
  {
      "question": "<natural language question>",
      "user_intent": "<intent type>",
      "schema": <structured intent subset>
  }

Language
--------
- Brazilian Portuguese (pt-BR) only

Notes
-----
- This pipeline assumes all intents are already valid and cleaned.
- No semantic validation or quality scoring is performed here.
- Generated questions are concise, natural, and conversational,
  avoiding any schema- or database-specific terminology.
"""

import json
import os
import time
from typing import Any, Dict, List, ClassVar

from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts, Step
from distilabel.steps import StepInput, StepOutput
from distilabel.steps.tasks import TextGeneration
from distilabel.models.llms import GroqLLM

from config.paths import CLEANED_INTENTS_DIR, BASE_DATASETS_DIR
from config.env_loader import load_env


# ============================================================
# Environment setup
# ============================================================

# Load environment variables (API keys, tokens, etc.)
# This must be executed before initializing any LLM backend
load_env()


# ============================================================
# Utility functions
# ============================================================

def load_intents(path: str) -> List[Dict[str, Any]]:
    """
    Load structured intents from a JSONL file.

    Parameters
    ----------
    path : str
        Path to the JSONL file containing cleaned intents.

    Returns
    -------
    List[Dict[str, Any]]
        List of intent dictionaries.
    """
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split a list into fixed-size batches.

    This is used to control LLM request size and
    avoid rate limits or memory issues.

    Parameters
    ----------
    lst : list
        Input list to be chunked.
    chunk_size : int
        Maximum number of elements per chunk.

    Returns
    -------
    List[List[Any]]
        List of chunks.
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def safe_value(obj: Dict[str, Any]) -> str:
    """
    Extract an explicitly defined value from a constraint object.

    Supports multiple value types and legacy schemas.
    Falls back to a generic placeholder only if no value is provided.

    Parameters
    ----------
    obj : Dict[str, Any]
        Constraint object from the intent schema.

    Returns
    -------
    str
        String representation of the constraint value.
    """
    if obj.get("value_str") is not None:
        return str(obj["value_str"])
    if obj.get("value_int") is not None:
        return str(obj["value_int"])
    if obj.get("value_float") is not None:
        return str(obj["value_float"])
    if obj.get("value") is not None:  # legacy support
        return str(obj["value"])
    return "<VALUE>"


def intent_to_text(intent: dict) -> str:
    """
    Convert a structured graph intent into a textual
    description suitable for LLM prompting.

    This representation is NOT shown to the final user.
    It exists only to guide controlled question generation.

    Parameters
    ----------
    intent : dict
        Structured intent specification.

    Returns
    -------
    str
        Textual description of the intent.
    """
    parts = [f"User intent: {intent.get('user_intent', 'unknown')}"]

    # Describe traversal path if present
    if intent.get("path"):
        path_desc = " -> ".join(
            f"{p.get('from')} {p.get('rel')} {p.get('to')}"
            for p in intent["path"]
        )
        parts.append(f"Path: {path_desc}")

    # Describe known entities (anchors)
    if intent.get("known"):
        known_desc = ", ".join(
            f"{k.get('label')} {k.get('attribute')} "
            f"{k.get('operator')} {safe_value(k)}"
            for k in intent["known"]
        )
        parts.append(f"Known nodes: {known_desc}")

    # Describe filtering constraints
    filters = intent.get("constraints", {}).get("filters", [])
    if filters:
        filters_desc = ", ".join(
            f"{f.get('label')} {f.get('attribute')} "
            f"{f.get('operator')} {safe_value(f)}"
            for f in filters
        )
        parts.append(f"Filters: {filters_desc}")

    # Describe return projection
    if intent.get("return"):
        ret = intent["return"]
        attrs = ", ".join(ret.get("attributes", []))
        parts.append(f"Return attributes: {ret.get('label')} ({attrs})")

    return "\n".join(parts)


# ============================================================
# Pipeline Step: Intent → Instruction
# ============================================================

class IntentToInstruction(Step):
    """
    Convert a structured graph intent into a SYSTEM instruction
    that prompts an LLM to generate a single natural language question.

    This step does NOT generate the question itself.
    It only produces a controlled instruction that enforces:

    - Language constraints (Brazilian Portuguese)
    - Conversational and natural phrasing
    - Strict semantic faithfulness to the intent
    - No inference or hallucination beyond provided schema
    """

    outputs: ClassVar[list[str]] = ["instruction"]

    def process(self, inputs: StepInput) -> StepOutput:
        outputs = []

        for intent in inputs:
            if not isinstance(intent, dict):
                continue

            if not intent.get("user_intent") or not intent.get("return"):
                continue

            intent_text = intent_to_text(intent)

            instruction = (
                "[SYSTEM]\n"
                "You are an expert assistant for graph databases.\n"
                "Convert a structured query intent into ONE natural question.\n\n"
                "RULES:\n"
                "- Write strictly in Brazilian Portuguese (pt-BR)\n"
                "- Use simple, conversational language\n"
                "- Be concise and objective\n"
                "- Do NOT explain data structures\n"
                "- Do NOT add or infer information\n"
                "- Return ONLY one question\n\n"
                "[USER]\n"
                f"{intent_text}"
            )

            outputs.append(
                {
                    **intent,
                    "instruction": instruction,
                }
            )

        if outputs:
            yield outputs


# ============================================================
# Pipeline Step: Question + Schema Selection
# ============================================================

class SelectQuestionSchema(Step):
    """
    Select and normalize the final dataset example by extracting
    the generated natural language question and attaching the
    corresponding structured intent schema.

    This step ensures a clean and explicit mapping between:
    - Natural language question
    - User intent type
    - Minimal schema representation required for supervision
    """

    outputs: ClassVar[list[str]] = ["question", "user_intent", "schema"]

    def process(self, items: StepInput) -> StepOutput:
        results = []

        for item in items:
            question = item.get("qa_pair")
            if not isinstance(question, str):
                continue

            schema = {
                "query_pattern": item.get("query_pattern"),
                "path": item.get("path"),
                "constraints": item.get("constraints"),
                "known": item.get("known"),
                "return": item.get("return"),
            }

            results.append(
                {
                    "question": question.strip(),
                    "user_intent": item.get("user_intent"),
                    "schema": schema,
                }
            )

        if results:
            yield results


# ============================================================
# Entry point
# ============================================================

def main() -> None:
    """
    Execute the full question dataset generation pipeline.

    This function:
    - Loads cleaned intents from disk
    - Processes them in small batches to control LLM usage
    - Generates one natural language question per intent
    - Persists the final question + schema dataset as JSONL
    """

    intents_path = os.path.join(CLEANED_INTENTS_DIR, "intents_clean.jsonl")
    output_path = os.path.join(BASE_DATASETS_DIR, "questions_base.jsonl")

    batch_size = 2

    intents = load_intents(intents_path)
    print(f"Loaded {len(intents)} intents")

    all_outputs: List[Dict[str, Any]] = []

    # Process intents in small batches to control
    # LLM latency, cost and rate limits
    for batch_idx, batch in enumerate(chunk_list(intents, batch_size)):
        print(f"\nProcessing batch {batch_idx + 1}")

        with Pipeline(name=f"question-batch-{batch_idx}") as pipeline:

            load_step = LoadDataFromDicts(
                name="load_intents",
                data=batch,
            )

            intent_step = IntentToInstruction(
                name="intent_to_instruction"
            )

            generation_step = TextGeneration(
                name="generate_question",
                llm=GroqLLM(
                    model="llama-3.1-8b-instant",
                    generation_kwargs={"temperature": 0.3},
                ),
                columns=["instruction"],
                output_mappings={"generation": "qa_pair"},
            )

            select_step = SelectQuestionSchema(
                name="select_question_schema"
            )

            # Define execution graph
            load_step >> intent_step >> generation_step >> select_step

        dataset = pipeline.run(use_cache=False)
        all_outputs.extend(dataset["default"]["train"].to_list())

        # Small pause to reduce API pressure
        time.sleep(2)

    # --------------------------------------------------
    # Persist final dataset
    # --------------------------------------------------
    with open(output_path, "w", encoding="utf-8") as f:
        for row in all_outputs:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")

    print(f"Dataset saved to: {output_path}")


if __name__ == "__main__":
    main()
