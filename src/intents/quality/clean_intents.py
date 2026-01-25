"""
Intent Cleaning Runner
======================

Entry-point script for executing the intent cleaning stage of the
dataset generation pipeline.

This step applies deterministic quality scoring to previously
validated intents, filtering out low-quality or ambiguous examples
based on a minimum quality score threshold.

Accepted intents are persisted as the final clean dataset, while
rejected ones may optionally be stored for inspection and analysis.

Dependencies
------------
- pathlib
- intent_quality_filter
- config.paths

Usage
-----
Run the script directly from the command line:

    python clean_intents.py

Notes
-----
- This script assumes intents were already semantically validated.
- Quality filtering is deterministic and explainable.
- Rejected intents can be stored for auditing or threshold tuning.
"""

from pathlib import Path

from .intent_quality_filter import IntentCleaner
from config.paths import VALIDATED_INTENTS_DIR, CLEANED_INTENTS_DIR


# --------------------------------------------------
# Entry point
# --------------------------------------------------

def main() -> None:
    """
    Execute the intent quality cleaning pipeline.
    """

    # --------------------------------------------------
    # Configuration
    # --------------------------------------------------
    min_score = 0.7  # heuristic threshold based on qualitative inspection
    keep_diagnostics = False  # enable for debugging / auditing

    # --------------------------------------------------
    # Define input and output paths
    # --------------------------------------------------
    input_path = Path(VALIDATED_INTENTS_DIR) / "intents_valid.jsonl"

    output_path = Path(CLEANED_INTENTS_DIR) / "intents_clean.jsonl"
    rejected_path = Path(CLEANED_INTENTS_DIR) / "intents_rejected.jsonl"

    # --------------------------------------------------
    # Initialize intent cleaner
    # --------------------------------------------------
    cleaner = IntentCleaner(min_score=min_score)

    # --------------------------------------------------
    # Execute cleaning pipeline
    # --------------------------------------------------
    cleaner.run(
        input_path=input_path,
        output_path=output_path,
        rejected_path=rejected_path,
        keep_diagnostics=keep_diagnostics,
    )


if __name__ == "__main__":
    main()
