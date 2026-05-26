"""
Question Generation Pipeline
============================

Deterministic natural-language generation pipeline for structured
graph query intents.

This module converts validated graph query intents into realistic
natural-language questions in Brazilian Portuguese (pt-BR) while
preserving semantic fidelity and structural recoverability.

Each structured intent is transformed into exactly one natural
question aligned with its underlying graph-query semantics.

Primary Goals
-------------
- preserve structural intent semantics
- maintain schema recoverability
- generate realistic user questions
- prevent hallucinated information
- support deterministic dataset generation
- ensure compatibility with semantic parsing tasks

Generated datasets are suitable for:
- LLM fine-tuning
- instruction tuning
- semantic parsing
- intent understanding
- NL-to-graph-query mapping
- graph question answering research

Key Characteristics
-------------------
- generates exactly one question per intent
- deterministic prompting strategy
- schema-grounded generation
- no Cypher or query generation
- no explanations or reasoning output
- semantic fidelity prioritized over fluency
- optimized for graph-query datasets

Input
-----
JSONL file containing structurally validated intents.

Expected intent structure:
{
    "intent": {...},
    "schema_spec": {...}
}

Output
------
JSONL dataset containing:
{
    "question": "<pt-BR natural language question>",
    "regime": "<structural regime>",
    "schema": <normalized schema>
}

Language
--------
Brazilian Portuguese (pt-BR) only.

Notes
-----
This module assumes all intents are already:
- structurally validated
- semantically validated
- filtered by generation policies

No validation or schema correction is performed here.
"""

import json
import os
import time
import yaml
from typing import Any, Dict, List, ClassVar
from tqdm import tqdm

from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts, Step
from distilabel.steps import StepInput, StepOutput
from distilabel.steps.tasks import TextGeneration
from distilabel.models.llms import GroqLLM

from config.paths import INTENTS_DATA_DIR, BASE_DATASETS_DIR, DATASETS_CONFIG_DIR
from config.env_loader import load_env


# ============================================================
# Environment setup
# ============================================================

# Load environment variables (API keys, etc.)
# Must be executed before initializing the LLM client
load_env()


# ============================================================
# Configuration
# ============================================================

# Centralized configuration ensures reproducibility
CONFIG_PATH = os.path.join(
    DATASETS_CONFIG_DIR, "generation_config.yaml"
)

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg: Dict[str, Any] = yaml.safe_load(f)


# LLM generation parameters
MODEL_NAME = cfg["generation"]["model"]
TEMPERATURE = cfg["generation"]["temperature"]
BATCH_SIZE = cfg["generation"]["batch_size"]
SLEEP_SECONDS = cfg["generation"]["sleep_seconds"]

# dataset paths
INTENTS_FILE = cfg["input"]["intents_file"]
OUTPUT_FILE = cfg["output"]["dataset_file"]


# ============================================================
# Utility functions
# ============================================================

def load_intents(path: str) -> List[Dict[str, Any]]:
    """
    Load structured intents from a JSONL dataset.

    Each line must contain a single validated structured
    graph-query intent.

    Parameters
    ----------
    path : str
        Path to the JSONL file.

    Returns
    -------
    List[Dict[str, Any]]
        Loaded structured intents.
    """

    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split a list into fixed-size batches.

    Batch processing improves:
    - API stability
    - token usage control
    - memory efficiency
    - generation robustness
    - rate-limit handling

    Parameters
    ----------
    lst : List[Any]
        Input list.

    chunk_size : int
        Maximum batch size.

    Returns
    -------
    List[List[Any]]
        List of batches.
    """

    return [
        lst[i:i + chunk_size]
        for i in range(0, len(lst), chunk_size)
    ]


def intent_to_text(intent: dict) -> str:
    """
    Convert a structured intent into a deterministic semantic
    intermediate representation for LLM prompting.

    The generated representation exposes structural query
    semantics explicitly while remaining easy for language
    models to verbalize into natural questions.

    The representation preserves:
    - regime semantics
    - traversal structure
    - filter ownership
    - aggregation behavior
    - ordering semantics
    - limiting constraints
    - return attributes

    Parameters
    ----------
    intent : dict
        Structured graph-query intent.

    Returns
    -------
    str
        Deterministic semantic representation used for prompting.
    """

    intent_meta = intent.get("intent", {})
    schema = intent.get("schema_spec", {})

    lines = []

    # ---------------------------------------------------------
    # Query regime
    # ---------------------------------------------------------

    regime = intent_meta.get("regime", "unknown")

    lines.append("Query regime:")
    lines.append(f"- {regime}")

    # ---------------------------------------------------------
    # Target entity
    # ---------------------------------------------------------

    target = schema.get("target", {}).get("label")

    if target:
        lines.append("")
        lines.append("Target entity:")
        lines.append(f"- {target}")

    # ---------------------------------------------------------
    # Relationship path
    # ---------------------------------------------------------

    path = schema.get("path") or []

    if path:

        lines.append("")
        lines.append("Relationships:")

        for step in path:

            source = (
                step.get("source", {})
                .get("label", "Unknown")
            )

            target_node = (
                step.get("target", {})
                .get("label", "Unknown")
            )

            relation_text = (
                f"- {target_node} "
                f"located inside "
                f"{source}"
            )

            lines.append(relation_text)

    # ---------------------------------------------------------
    # Filters
    # ---------------------------------------------------------

    filters = schema.get("filters") or []

    if filters:

        lines.append("")
        lines.append("Conditions:")

        for f in filters:

            node = f.get("node_label")
            attr = f.get("attribute")

            op = normalize_operator(
                f.get("operator")
            )

            value = safe_value(f)

            if attr == "type" and f.get("operator") == "=":

                lines.append(
                    f"- {node} has type {value}"
                )

            else:

                lines.append(
                    f"- {node} {attr} {op} {value}"
                )

    # ---------------------------------------------------------
    # Aggregation
    # ---------------------------------------------------------

    aggregate = schema.get("aggregate")

    if aggregate:

        function = aggregate.get("function")
        attribute = aggregate.get("attribute")

        lines.append("")
        lines.append("Aggregation:")

        if attribute:

            lines.append(
                f"- {function} of {attribute}"
            )

        else:

            lines.append(
                f"- {function}"
            )

    # ---------------------------------------------------------
    # Ordering
    # ---------------------------------------------------------

    order_by = schema.get("order_by")

    if order_by:

        lines.append("")
        lines.append("Sorting:")

        attribute = order_by.get("attribute")

        direction = order_by.get(
            "direction",
            "asc"
        )

        lines.append(
            f"- {attribute} {direction}"
        )

    # ---------------------------------------------------------
    # Limit
    # ---------------------------------------------------------

    limit = schema.get("limit")

    if limit is not None:

        lines.append("")
        lines.append("Limit:")
        lines.append(f"- {limit}")

    # ---------------------------------------------------------
    # Return attributes
    # ---------------------------------------------------------

    returns = schema.get("return_attributes") or []

    if returns:

        lines.append("")
        lines.append("Return:")
        lines.append(
            f"- {', '.join(returns)}"
        )

    return "\n".join(lines)


def normalize_operator(operator: str) -> str:
    """
    Convert symbolic operators into deterministic semantic text.

    Examples
    --------
    >  -> greater than
    <= -> less than or equal to
    =  -> equal to

    Parameters
    ----------
    operator : str
        Symbolic operator.

    Returns
    -------
    str
        Semantic operator representation.
    """

    mapping = {
        ">": "greater than",
        ">=": "greater than or equal to",
        "<": "less than",
        "<=": "less than or equal to",
        "=": "equal to",
        "contains": "contains"
    }

    return mapping.get(operator, operator)


def safe_value(obj: Dict[str, Any]) -> str:
    """
    Extract filter values deterministically from schema filters.

    Supports both legacy and normalized schema formats while
    preserving backward compatibility.

    Priority order:
    - value_str
    - value_int
    - value_float
    - value

    Parameters
    ----------
    obj : Dict[str, Any]
        Filter specification.

    Returns
    -------
    str
        Deterministic string representation of the value.
    """

    priority_keys = [
        "value_str",
        "value_int",
        "value_float",
        "value"
    ]

    for key in priority_keys:

        value = obj.get(key)

        if value is not None:
            return str(value)

    return "<VALUE>"


# ============================================================
# Pipeline Step: Intent → Instruction
# ============================================================

class IntentToInstruction(Step):
    """
    Convert structured intents into controlled generation prompts.

    This step transforms validated graph-query intents into
    deterministic instructions optimized for natural-language
    question generation.

    The generated prompts constrain the LLM to:
    - preserve semantic fidelity
    - preserve structural recoverability
    - avoid hallucinations
    - maintain relationship ownership
    - preserve filters and operators
    - generate exactly one question

    The resulting prompts are intentionally restrictive to
    maximize consistency and dataset quality.
    """

    outputs: ClassVar[list[str]] = ["instruction"]

    def process(self, inputs: StepInput) -> StepOutput:
        """
        Convert structured intents into controlled LLM instructions.

        Processing stages:
        - validate input structure
        - generate semantic intermediate representation
        - construct deterministic generation prompt
        - attach instruction to pipeline item

        Parameters
        ----------
        inputs : StepInput
            Batch of structured intents.

        Returns
        -------
        StepOutput
            Batch containing generation instructions.
        """

        outputs = []

        for intent in inputs:

            # -------------------------------------------------
            # Skip malformed examples
            # -------------------------------------------------

            if not isinstance(intent, dict):
                continue

            if (
                "intent" not in intent
                or "schema_spec" not in intent
            ):
                continue

            # -------------------------------------------------
            # Convert schema into semantic IR
            # -------------------------------------------------

            intent_text = intent_to_text(intent)

            # -------------------------------------------------
            # Controlled generation prompt
            # -------------------------------------------------

            instruction = (
                "[SYSTEM]\n"
                "Convert the structured intent into EXACTLY ONE natural language question in Brazilian Portuguese.\n\n"

                "LANGUAGE:\n"
                "- Write ONLY in pt-BR\n\n"

                "PRIMARY GOAL:\n"
                "- Preserve the exact semantic structure\n"
                "- Preserve full structural recoverability\n"
                "- Include ALL filters, relationships, aggregations, ordering, and limits\n"
                "- Keep every condition attached to the correct entity\n\n"

                "STRICT RULES:\n"
                "- NEVER add information\n"
                "- NEVER omit information\n"
                "- NEVER infer hidden meaning\n"
                "- NEVER use world knowledge\n"
                "- NEVER reinterpret entities\n"
                "- NEVER reinterpret relationships\n"
                "- NEVER move conditions between entities\n"
                "- NEVER introduce concepts not explicitly present\n"
                "- NEVER summarize filters\n"
                "- NEVER replace operators with vague language\n"
                "- NEVER invert comparison operators\n"
                "- NEVER replace ordering semantics\n"
                "- NEVER introduce popularity, relevance, importance, size, or quality semantics\n"
                "- Prefer semantic fidelity over fluency\n"
                "- Repetition is acceptable if necessary\n\n"

                "ENTITY RULES:\n"
                "- The target entity MUST be the main subject of the question\n"
                "- Preserve entity associations explicitly\n"
                "- Preserve relationship traversal semantics exactly\n\n"

                "FILTER RULES:\n"
                "- Preserve numeric comparisons exactly\n"
                "- Preserve comparison direction exactly\n"
                "- Preserve attribute semantics exactly\n\n"

                "AGGREGATION MAPPINGS:\n"
                "- count -> quantos\n"
                "- avg -> média\n"
                "- sum -> soma total\n"
                "- max -> maior\n"
                "- min -> menor\n\n"

                "ORDERING RULES:\n"
                "- asc -> ordem crescente\n"
                "- desc -> ordem decrescente\n"
                "- If ordering by name, explicitly mention alphabetical ordering\n"
                "- Do NOT replace ordering with ranking or popularity semantics\n\n"

                "LIMIT RULES:\n"
                "- Express limits explicitly when present\n"
                "- Example: 'os 10 primeiros'\n\n"

                "STYLE:\n"
                "- Write a natural but structurally faithful question\n"
                "- Avoid unnecessary creativity\n"
                "- Avoid database terminology\n"
                "- Avoid excessive paraphrasing\n\n"

                "OUTPUT RULES:\n"
                "- Generate EXACTLY ONE question\n"
                "- Output ONLY the final question\n"
                "- No explanations\n"
                "- No JSON\n"
                "- No lists\n\n"

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
# Pipeline Step: Question + Schema normalization
# ============================================================

class SelectQuestionSchema(Step):
    """
    Extract generated questions and normalize schema structures.

    This step:
    - extracts the generated question
    - preserves regime metadata
    - normalizes filter values
    - prepares final dataset rows

    Schema normalization improves consistency for downstream:
    - semantic parsing
    - instruction tuning
    - structured prediction tasks
    """

    outputs: ClassVar[list[str]] = ["question", "regime", "schema"]

    def process(self, items: StepInput) -> StepOutput:
        """
        Extract final dataset rows from generation outputs.

        Processing stages:
        - extract generated question
        - recover regime metadata
        - normalize schema filter values
        - build final dataset structure

        Parameters
        ----------
        items : StepInput
            Generated pipeline outputs.

        Returns
        -------
        StepOutput
            Final normalized dataset rows.
        """

        results = []

        for item in items:

            question = item.get("qa_pair")

            if not isinstance(question, str):
                continue

            intent_meta = item.get("intent", {})
            schema = item.get("schema_spec", {})

            # -------------------------------------------------
            # Regime extraction
            # -------------------------------------------------

            regime = intent_meta.get("regime")

            # fallback for datasets where regime may exist
            # at top-level
            if regime is None:
                regime = item.get("regime")

            # normalize filter values for consistent dataset structure
            schema = self.normalize_filter_values(schema)

            results.append(
                {
                    "question": question.strip(),
                    "regime": regime,
                    "schema": schema,
                }
            )

        if not results:
            print("WARNING: empty batch in SelectQuestionSchema")

        yield results

    @staticmethod
    def normalize_filter_values(schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize filter values into explicit typed fields.

        Converts generic filter values into deterministic typed
        representations to reduce ambiguity during downstream
        training and evaluation.

        Generated fields:
        - value_str
        - value_int
        - value_float

        Parameters
        ----------
        schema : Dict[str, Any]
            Schema specification.

        Returns
        -------
        Dict[str, Any]
            Normalized schema.
        """

        filters = schema.get("filters", [])

        for f in filters:

            value = f.pop("value", None)

            # explicit typing avoids ambiguity for downstream models
            f["value_str"] = None
            f["value_int"] = None
            f["value_float"] = None

            if isinstance(value, str):
                f["value_str"] = value

            elif isinstance(value, int):
                f["value_int"] = value

            elif isinstance(value, float):
                f["value_float"] = value

            elif value is not None:
                f["value_str"] = str(value)

        return schema


# ============================================================
# Entry point
# ============================================================

def main() -> None:
    """
    Execute the full natural-language dataset generation pipeline.

    Workflow
    --------
    1. load validated intents
    2. batch intents
    3. convert intents into instructions
    4. generate questions using LLM
    5. normalize schema values
    6. export dataset as JSONL

    Batch execution is used to:
    - improve robustness
    - reduce API instability
    - control token usage
    - prevent rate-limit spikes

    Output
    ------
    JSONL dataset containing:
    - generated questions
    - structural regimes
    - normalized schema definitions
    """

    intents_path = os.path.join(INTENTS_DATA_DIR, INTENTS_FILE)
    output_path = os.path.join(BASE_DATASETS_DIR, OUTPUT_FILE)

    intents = load_intents(intents_path)

    print(f"Loaded {len(intents)} intents")

    all_outputs: List[Dict[str, Any]] = []

    batches = chunk_list(intents, BATCH_SIZE)

    print(f"Total intents: {len(intents)}")
    print(f"Total batches: {len(batches)}")

    for batch_idx, batch in enumerate(
        tqdm(batches, desc="Generating questions", unit="batch")
    ):

        # each batch runs as an independent distilabel pipeline
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
                    model=MODEL_NAME,
                    generation_kwargs={
                        "temperature": TEMPERATURE,
                    },
                ),

                # only the instruction column is sent to the LLM
                columns=["instruction"],

                output_mappings={
                    "generation": "qa_pair"
                },
            )

            select_step = SelectQuestionSchema(
                name="select_question_schema"
            )

            load_step >> intent_step >> generation_step >> select_step

        dataset = pipeline.run(use_cache=False)

        all_outputs.extend(
            dataset["default"]["train"].to_list()
        )

        # small delay prevents API rate spikes
        time.sleep(SLEEP_SECONDS)

    with open(output_path, "w", encoding="utf-8") as f:

        for row in all_outputs:

            json.dump(row, f, ensure_ascii=False)

            f.write("\n")

    print(f"Dataset saved to: {output_path}")


if __name__ == "__main__":

    main()
