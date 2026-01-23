"""
Intent Derivation Pipeline
==========================

Pipeline responsible for deriving structured intent specifications
from a predefined graph schema and exporting them as a JSONL dataset.

This module orchestrates schema loading and controlled combinatorial
intent generation using the IntentGenerator, producing a bounded and
semantically valid intent corpus.

The generated intents are designed for downstream stages such as:
- Semantic validation
- Quality scoring and filtering
- Instruction and question generation
- Dataset construction for LLM fine-tuning and evaluation

This module performs *no linguistic generation* and focuses strictly
on canonical, machine-readable intent representations.

Important Characteristics
-------------------------
- Schema-driven and domain-agnostic
- Deterministic generation
- Explicit bounds to prevent combinatorial explosion
- Produces normalized JSONL output

Dependencies
------------
- json
- pathlib
- typing
- config.paths
- schema.schema_loader
- intents.generation.intent_generator

Usage
-----
Run the pipeline directly from the command line:

    python intent_derivation.py

Notes
-----
- The output JSONL file is overwritten if it already exists.
- All generated intents conform to the normalized dataset schema.
"""

import json
from pathlib import Path
from typing import List

from config.paths import SCHEMA_DATA_DIR, RAW_INTENTS_DIR
from schema.schema_loader import load_schema
from .intent_generator import IntentGenerator


# --------------------------------------------------
# Serialization utilities
# --------------------------------------------------

def export_intents_jsonl(intents: List[dict], output_path: Path) -> None:
    """
    Export structured intent specifications to a JSON Lines (JSONL) file.

    Each intent is written as a single JSON object per line.

    Parameters
    ----------
    intents : List[dict]
        List of normalized intent dictionaries.
    output_path : Path
        Destination path for the JSONL output file.

    Notes
    -----
    - Assumes the parent directory already exists.
    - Existing files at the destination path are overwritten.
    """

    with output_path.open("w", encoding="utf-8") as f:
        for intent in intents:
            f.write(json.dumps(intent, ensure_ascii=False))
            f.write("\n")


# --------------------------------------------------
# Main execution
# --------------------------------------------------

if __name__ == "__main__":

    # --------------------------------------------------
    # Load graph schema definition
    # --------------------------------------------------
    schema_path = Path(SCHEMA_DATA_DIR) / "graph_schema.json"
    schema = load_schema(schema_path)

    # --------------------------------------------------
    # Initialize intent generator with safety bounds
    # --------------------------------------------------
    generator = IntentGenerator(
        schema=schema,
        max_variants=15,
        max_two_hop_intents=60,
        max_two_hop_filtered_intents=50,
    )

    # --------------------------------------------------
    # Generate normalized intent dataset
    # --------------------------------------------------
    intents = generator.generate_dataset()
    print(f"Generated {len(intents)} intents")

    # --------------------------------------------------
    # Inspect sample intents (sanity check)
    # --------------------------------------------------
    for intent in intents[:5]:
        print(intent)

    # --------------------------------------------------
    # Persist dataset to JSONL
    # --------------------------------------------------
    output_path = Path(RAW_INTENTS_DIR) / "intents_raw.jsonl"
    export_intents_jsonl(intents, output_path)

    print(f"Intents exported to: {output_path}")
