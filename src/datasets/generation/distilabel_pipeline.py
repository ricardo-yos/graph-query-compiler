"""
Question Generation Pipeline
============================

Deterministic pipeline that converts structured graph query intents
into natural language questions in Brazilian Portuguese (pt-BR).

Each structured intent is transformed into exactly one realistic
user question while preserving the original schema for full
traceability between natural language and graph query structure.

The resulting dataset is suitable for:
- LLM fine-tuning
- instruction-following alignment
- semantic parsing
- intent understanding
- natural language → graph query mapping

Key Characteristics
-------------------
- Generates only questions (no answers, no Cypher, no explanations)
- Strictly grounded on the provided intent schema
- Deterministic prompting for consistency and reproducibility
- Preserves semantic fidelity to the original structure
- Avoids hallucination by restricting generation to schema information
- Optimized for realistic user queries over graph-based systems

Input
-----
JSONL file containing structurally validated intents.

Output
------
JSONL dataset where each row contains:

{
    "question": "<natural language question>",
    "user_intent": "<intent type>",
    "schema": <structured intent subset>
}

Language
--------
Brazilian Portuguese (pt-BR) only.

Notes
-----
This pipeline assumes all intents are already validated and filtered.
No structural or semantic validation is performed here.
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
    DATASETS_CONFIG_DIR, "generation.yaml"
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
    Load structured intents from a JSONL file.

    Each line must contain one JSON object describing
    a validated intent structure.

    Parameters
    ----------
    path : str
        Path to the JSONL file.

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

    Batching helps:
    - control token usage
    - reduce latency spikes
    - avoid API rate limits
    - improve pipeline stability

    Parameters
    ----------
    lst : List[Any]
        Input list.
    chunk_size : int
        Maximum number of elements per batch.

    Returns
    -------
    List[List[Any]]
        List of batches.
    """

    return [
        lst[i:i + chunk_size]
        for i in range(0, len(lst), chunk_size)
    ]


def safe_value(obj: Dict[str, Any]) -> str:
    """
    Extract filter value from multiple schema formats.

    Supports backward compatibility with legacy datasets
    where value typing may differ.

    Parameters
    ----------
    obj : Dict[str, Any]
        Filter specification.

    Returns
    -------
    str
        String representation of the filter value.
    """

    if obj.get("value_str") is not None:
        return str(obj["value_str"])

    if obj.get("value_int") is not None:
        return str(obj["value_int"])

    if obj.get("value_float") is not None:
        return str(obj["value_float"])

    if obj.get("value") is not None:
        return str(obj["value"])

    # placeholder keeps sentence structure valid
    return "<VALUE>"


def intent_to_text(intent: dict) -> str:
    """
    Convert structured intent into a controlled textual representation.

    This representation acts as an intermediate format that helps the LLM:

    - preserve schema semantics
    - maintain structural fidelity
    - avoid hallucinating unsupported information
    - produce consistent question patterns

    Parameters
    ----------
    intent : dict
        Structured intent specification.

    Returns
    -------
    str
        Text describing the intent semantics.
    """

    intent_meta = intent.get("intent", {})
    schema = intent.get("schema_spec", {})

    parts = []

    # intent category influences wording style
    intent_type = intent_meta.get("type", "unknown")
    modifiers = intent_meta.get("modifiers") or []

    parts.append(f"Intent type: {intent_type}")

    if modifiers:
        parts.append(
            f"Modifiers: {', '.join(modifiers)}"
        )

    # main entity queried by the user
    target = schema.get("target", {}).get("label")

    if target:
        parts.append(
            f"Target node: {target}"
        )

    # traversal path between entities
    path = schema.get("path")

    if path:
        path_desc = " -> ".join(

            f"{step.get('relationship')} -> {step.get('target')}"

            for step in path
        )

        parts.append(
            f"Traversal path: {path_desc}"
        )

    # constraints applied to nodes
    filters = schema.get("filters")

    if filters:

        filters_desc = ", ".join(

            f"{f.get('node_label')} "
            f"{f.get('attribute')} "
            f"{f.get('operator')} "
            f"{safe_value(f)}"

            for f in filters
        )

        parts.append(
            f"Filters: {filters_desc}"
        )

    # attributes expected in the result
    returns = schema.get("return_attributes")

    if returns:

        parts.append(
            f"Return attributes: {', '.join(returns)}"
        )

    # ranking influences phrasing like "top 10"
    order_by = schema.get("order_by")

    if order_by:

        order_desc = ", ".join(

            f"{o.get('attribute')} "
            f"{o.get('direction','ASC')}"

            for o in order_by
        )

        parts.append(
            f"Order by: {order_desc}"
        )

    # limit affects expressions like "first 5"
    limit = schema.get("limit")

    if limit:

        parts.append(
            f"Limit: {limit}"
        )

    # aggregation affects semantics strongly
    aggregate = schema.get("aggregate")

    if aggregate:

        parts.append(

            f"Aggregate: "
            f"{aggregate.get('function')} "
            f"{aggregate.get('attribute')}"

        )

    return "\n".join(parts)


# ============================================================
# Pipeline Step: Intent → Instruction
# ============================================================

class IntentToInstruction(Step):
    """
    Convert structured intents into deterministic instructions
    for the LLM.

    The generated instruction constrains the model to:

    - produce one question only
    - preserve semantic meaning of the schema
    - avoid implementation-specific terminology
    - write fluent Brazilian Portuguese
    """

    outputs: ClassVar[list[str]] = ["instruction"]

    def process(self, inputs: StepInput) -> StepOutput:

        outputs = []

        for intent in inputs:

            # skip malformed records
            if not isinstance(intent, dict):
                continue

            if "intent" not in intent or "schema_spec" not in intent:
                continue

            intent_text = intent_to_text(intent)

            # highly constrained prompt improves consistency
            # and reduces hallucination risk
            instruction = (
                "[SYSTEM]\n"
                "You convert a structured graph intent into ONE natural question.\n\n"

                "Write the question in Brazilian Portuguese (pt-BR).\n\n"

                "Goal:\n"
                "Create 1 natural question that expresses exactly the information contained in the intent.\n\n"

                "GENERAL:\n"
                "- The question must sound natural and realistic\n"
                "- Use conversational language\n"
                "- Reflect the full meaning of the intent\n"
                "- Do not add information not present in the schema\n"
                "- Do not omit relevant information\n"
                "- Prefer short but complete questions\n\n"

                "TARGET:\n"
                "- The target label must appear as the main subject of the question\n"
                "- The user should clearly be asking about this entity type\n\n"

                "PATH:\n"
                "- Represent each relationship as a natural phrase connecting entities\n"
                "- Preserve the logical order of relationships\n"
                "- The path should appear as contextual information in the question\n\n"

                "FILTERS:\n"
                "- All conditions must appear explicitly in the question\n"
                "- Include attribute values when present\n"
                "- Use natural wording instead of attribute names\n\n"

                "AGGREGATE:\n"
                "- Express aggregation clearly using natural language\n"
                "  avg → média\n"
                "  count → quantos\n"
                "  max → maior\n"
                "  min → menor\n\n"

                "ORDER:\n"
                "- Express ranking naturally\n"
                "  desc → maiores, melhores, mais bem avaliados\n"
                "  asc → menores, mais baratos\n\n"

                "LIMIT:\n"
                "- Express numeric limits explicitly when present\n\n"

                "CONSISTENCY:\n"
                "- The question should allow reconstruction of the intent structure\n"
                "- Avoid ambiguity about which entity each condition refers to\n\n"

                "OUTPUT:\n"
                "- Generate only 1 question\n"
                "- Do not explain anything\n"
                "- Do not output JSON\n\n"

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
    Extract generated question and attach normalized schema.

    Ensures filter values have explicit typing, improving
    dataset consistency for training downstream models.
    """

    outputs: ClassVar[list[str]] = ["question", "user_intent", "schema"]

    def process(self, items: StepInput) -> StepOutput:

        results = []

        for item in items:

            question = item.get("qa_pair")

            if not isinstance(question, str):
                continue

            intent_meta = item.get("intent", {})
            schema = item.get("schema_spec", {})

            # normalize filter values for consistent dataset structure
            schema = self.normalize_filter_values(schema)

            results.append(
                {
                    "question": question.strip(),
                    "user_intent": intent_meta.get("type"),
                    "schema": schema,
                }
            )

        if results:
            yield results

    @staticmethod
    def normalize_filter_values(schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize filter values into explicit typed fields.

        Converts generic "value" fields into:
        - value_str
        - value_int
        - value_float

        Parameters
        ----------
        schema : Dict[str, Any]

        Returns
        -------
        Dict[str, Any]
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
    Execute the full dataset generation workflow.

    Workflow
    --------
    1. load structural intents
    2. convert intents to instructions
    3. generate one question per intent
    4. normalize schema values
    5. persist dataset as JSONL

    Batching is used to:

    - control API usage
    - improve robustness
    - prevent request bursts
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
                    generation_kwargs={"temperature": TEMPERATURE},
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
