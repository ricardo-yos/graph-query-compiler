"""
Intent Validation Pipeline
==========================

Pipeline stage responsible for validating and sanitizing generated intents
before they are passed to downstream components.

This script reads a JSONL file containing derived intents, applies semantic
and structural validation rules, and writes valid and invalid intents to
separate JSONL files.

This stage acts as a formal gatekeeper in the dataset generation pipeline,
ensuring that only intents aligned with natural language plausibility and
schema constraints are propagated forward.

Dependencies
------------
- json
- pathlib
- intent_validator
- config.paths

Usage
-----
Run the script directly from the command line:

    python validate_intents.py

Notes
-----
- Validation logic is delegated to `is_valid_intent`.
- Input and output files follow the JSON Lines (JSONL) format.
- Output files are overwritten if they already exist.
"""

import json
from pathlib import Path

from config.paths import RAW_INTENTS_DIR, VALIDATED_INTENTS_DIR
from .intent_validator import is_valid_intent


# --------------------------------------------------
# Intent validation utility
# --------------------------------------------------

def validate_intents(
    input_path: Path,
    valid_output_path: Path,
    invalid_output_path: Path,
) -> None:
    """
    Validate intents from a JSONL file and split them into valid
    and invalid outputs.

    Parameters
    ----------
    input_path : Path
        Path to the input JSONL file containing raw intents.
    valid_output_path : Path
        Path to the output JSONL file for validated intents.
    invalid_output_path : Path
        Path to the output JSONL file for rejected intents.

    Notes
    -----
    - Each line is expected to be a valid JSON object.
    - Validation criteria are defined in `is_valid_intent`.
    - This function performs no mutation or normalization;
      it only filters intents based on validity.
    """
    kept = 0
    removed = 0

    with (
        input_path.open("r", encoding="utf-8") as f_in,
        valid_output_path.open("w", encoding="utf-8") as f_valid,
        invalid_output_path.open("w", encoding="utf-8") as f_invalid,
    ):
        for line in f_in:
            line = line.strip()
            if not line:
                continue

            try:
                intent = json.loads(line)
            except json.JSONDecodeError:
                # Malformed JSON line
                f_invalid.write(line + "\n")
                removed += 1
                continue

            if is_valid_intent(intent):
                f_valid.write(json.dumps(intent, ensure_ascii=False) + "\n")
                kept += 1
            else:
                f_invalid.write(json.dumps(intent, ensure_ascii=False) + "\n")
                removed += 1

    print("=" * 40)
    print("Intent Validation Summary")
    print("=" * 40)
    print(f"Validated intents : {kept}")
    print(f"Rejected intents  : {removed}")


# --------------------------------------------------
# Entry point
# --------------------------------------------------

def main() -> None:
    """
    Execute the intent validation pipeline.
    """
    input_path = Path(RAW_INTENTS_DIR) / "intents_raw.jsonl"

    valid_output_path = Path(VALIDATED_INTENTS_DIR) / "intents_valid.jsonl"
    invalid_output_path = Path(VALIDATED_INTENTS_DIR) / "intents_invalid.jsonl"

    validate_intents(
        input_path=input_path,
        valid_output_path=valid_output_path,
        invalid_output_path=invalid_output_path,
    )


if __name__ == "__main__":
    main()
