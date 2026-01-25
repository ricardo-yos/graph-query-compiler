"""
Intent Quality Filter
=====================

Quality-based filtering stage for validated graph query intents.

This module applies deterministic quality scoring to intents that have
already passed semantic and structural validation. Intents below a
minimum quality threshold are rejected, ensuring that only expressive,
useful, and NL-friendly intents proceed to downstream stages such as:

- Instruction generation
- NL question synthesis
- Cypher query generation
- LLM fine-tuning datasets

This module does NOT perform semantic validation.
It assumes all intents are already valid.

Dependencies
------------
- json
- pathlib
- intent_quality_scorer

Usage
-----
This module is intended to be used as a pipeline stage:

    from intent_quality_filter import IntentCleaner

    cleaner = IntentCleaner(min_score=0.6)
    cleaner.run(
        input_path=...,
        output_path=...,
        rejected_path=...
    )

Notes
-----
- Quality metadata is used only for filtering decisions.
- Scores and diagnostic reasons are not persisted by default.
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple

from .intent_quality_scorer import IntentQualityScorer


# --------------------------------------------------
# Intent quality filtering stage
# --------------------------------------------------

class IntentCleaner:
    """
    Filters validated intents using deterministic quality scoring.

    Intents are scored according to structural richness, usefulness,
    and suitability for natural language and query generation.
    """

    def __init__(self, min_score: float = 0.6):
        """
        Parameters
        ----------
        min_score : float
            Minimum quality score required for an intent to be accepted.
        """
        self.min_score = min_score
        self.scorer = IntentQualityScorer()

    # --------------------------------------------------
    # I/O utilities
    # --------------------------------------------------

    def load(self, path: Path) -> List[dict]:
        """
        Load intents from a JSON Lines (JSONL) file.

        Parameters
        ----------
        path : Path
            Path to the input JSONL file.

        Returns
        -------
        List[dict]
            List of intent dictionaries.
        """
        intents: List[dict] = []

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                intents.append(json.loads(line))

        return intents

    def save(self, intents: List[dict], path: Path) -> None:
        """
        Save intents to a JSON Lines (JSONL) file.

        Parameters
        ----------
        intents : List[dict]
            Intents to be written.
        path : Path
            Destination file path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            for intent in intents:
                f.write(json.dumps(intent, ensure_ascii=False) + "\n")

    # --------------------------------------------------
    # Quality filtering logic
    # --------------------------------------------------

    def clean(
        self,
        intents: List[dict],
        keep_diagnostics: bool = False,
    ) -> Tuple[List[dict], List[dict]]:
        """
        Score intents and split them into accepted and rejected sets.

        Parameters
        ----------
        intents : List[dict]
            Validated intents to be scored.
        keep_diagnostics : bool, optional
            If True, rejected intents will include score and rejection reasons.

        Returns
        -------
        accepted : List[dict]
            Intents that meet or exceed the quality threshold.
        rejected : List[dict]
            Intents rejected due to low quality.

        Important
        ---------
        - Quality metadata is NOT added to accepted intents.
        - Scoring is used strictly for filtering decisions.
        """
        accepted: List[dict] = []
        rejected: List[dict] = []

        for intent in intents:
            score, reasons = self.scorer.score(intent)

            if score >= self.min_score:
                accepted.append(intent)
            else:
                if keep_diagnostics:
                    rejected.append(
                        {
                            "intent": intent,
                            "score": score,
                            "reasons": reasons,
                        }
                    )
                else:
                    rejected.append(intent)

        return accepted, rejected

    # --------------------------------------------------
    # Pipeline orchestration
    # --------------------------------------------------

    def run(
        self,
        input_path: Path,
        output_path: Path,
        rejected_path: Optional[Path] = None,
        keep_diagnostics: bool = False,
    ) -> None:
        """
        Execute the full intent quality filtering workflow.

        Parameters
        ----------
        input_path : Path
            Path to the validated intents JSONL file.
        output_path : Path
            Path where accepted intents will be written.
        rejected_path : Optional[Path]
            Optional path for rejected intents.
        keep_diagnostics : bool
            Whether to include scoring diagnostics for rejected intents.
        """
        intents = self.load(input_path)

        accepted, rejected = self.clean(
            intents,
            keep_diagnostics=keep_diagnostics,
        )

        self.save(accepted, output_path)

        if rejected_path is not None:
            self.save(rejected, rejected_path)

        print("=" * 40)
        print("Intent Quality Filtering Summary")
        print("=" * 40)
        print(f"Accepted intents : {len(accepted)}")
        print(f"Rejected intents : {len(rejected)}")
        print(
            f"Acceptance rate  : "
            f"{len(accepted) / max(1, len(intents)):.2%}"
        )
