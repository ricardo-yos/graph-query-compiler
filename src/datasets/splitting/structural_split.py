"""
Structural Dataset Split
========================

Structure-aware train/validation split for datasets composed of
natural language questions paired with structured query schemas.

This module performs a feature-coverage split strategy that ensures
the training set contains all structural elements observed in the
dataset before allocating remaining samples to validation.

Structural features considered include:

- entity labels
- relationships
- attributes
- filter operators

Purpose
-------
Reduce the risk of evaluation failures caused by missing structural
elements in training while preserving a separate validation set for
model assessment.

Split Strategy
--------------
Phase 1
    Build a training set that covers the complete structural feature
    universe of the dataset.

Phase 2
    Adjust train/validation sizes according to the desired validation
    ratio while preserving structural coverage.

Input
-----
AUGMENTED_DATASETS_DIR / questions_paraphrased.jsonl

Output
------
SPLITS_DATASETS_DIR

├── train_base.jsonl
└── val_base.jsonl

Dependencies
------------
- json
- random
- pathlib
"""

import json
import random
from typing import Dict, List, Set, Tuple

from pathlib import Path
from config.paths import AUGMENTED_DATASETS_DIR, SPLITS_DATASETS_DIR


# ======================================================
# Feature Extraction
# ======================================================

def extract_features(schema: Dict) -> Dict[str, Set[str]]:
    """
    Extract structural features from a schema.

    Features are used to determine structural coverage and guide
    train/validation splitting.

    Extracted feature categories include:

    - entity labels
    - relationship types
    - attribute references
    - filter operators

    Parameters
    ----------
    schema : Dict
        Structured query schema.

    Returns
    -------
    Dict[str, Set[str]]
        Mapping of feature categories to extracted values.
    """
    features = {
        "labels": set(),
        "relations": set(),
        "attributes": set(),
        "operators": set(),
    }

    # Path traversal
    for p in schema.get("path", []):
        features["labels"].add(p.get("from"))
        features["labels"].add(p.get("to"))
        features["relations"].add(p.get("rel"))

    # Filters
    filters = schema.get("constraints", {}).get("filters", [])
    for f in filters:
        label = f.get("label")
        attr = f.get("attribute")
        op = f.get("operator")

        if label:
            features["labels"].add(label)
        if label and attr:
            features["attributes"].add(f"{label}.{attr}")
        if op:
            features["operators"].add(op)

    return features


# ======================================================
# Universe Construction
# ======================================================

def build_feature_universe(dataset: List[Dict]) -> Dict[str, Set[str]]:
    """
    Build the complete structural feature universe of a dataset.

    Aggregates all structural features observed across every schema
    in the dataset.

    Parameters
    ----------
    dataset : List[Dict]
        Dataset containing schema examples.

    Returns
    -------
    Dict[str, Set[str]]
        Complete set of observed structural features.
    """
    universe = {
        "labels": set(),
        "relations": set(),
        "attributes": set(),
        "operators": set(),
    }

    for item in dataset:
        feats = extract_features(item["schema"])
        for key in universe:
            universe[key].update(feats[key])

    return universe


# ======================================================
# Structural Split
# ======================================================

def structural_split(
    dataset: List[Dict],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Perform a structure-aware train/validation split.

    The algorithm first selects samples required to cover all
    structural features observed in the dataset.

    Remaining samples are then allocated to the validation set and
    adjusted to match the requested validation ratio.

    Parameters
    ----------
    dataset : List[Dict]
        Input dataset.

    val_ratio : float, default=0.2
        Desired validation proportion.

    seed : int, default=42
        Random seed used for reproducibility.

    Returns
    -------
    Tuple[List[Dict], List[Dict]]
        Train and validation datasets.
    """
    random.seed(seed)
    random.shuffle(dataset)

    covered = {
        "labels": set(),
        "relations": set(),
        "attributes": set(),
        "operators": set(),
    }

    train: List[Dict] = []
    val: List[Dict] = []

    def introduces_new_structure(features: Dict[str, Set[str]]) -> bool:
        """
        Check whether a sample introduces previously unseen
        structural features.

        Parameters
        ----------
        features : Dict[str, Set[str]]
            Features extracted from a schema.

        Returns
        -------
        bool
            True if at least one feature category contains values
            not yet covered by the training set.
        """
        return any(features[k] - covered[k] for k in covered)

    # Phase 1 — Ensure structural coverage
    for item in dataset:
        feats = extract_features(item["schema"])

        if introduces_new_structure(feats):
            train.append(item)
            for k in covered:
                covered[k].update(feats[k])
        else:
            val.append(item)

    # Phase 2 — Adjust validation size
    target_val_size = int(len(dataset) * val_ratio)

    if len(val) > target_val_size:
        excess = len(val) - target_val_size
        train.extend(val[:excess])
        val = val[excess:]

    elif len(val) < target_val_size:
        needed = target_val_size - len(val)
        moved = train[-needed:]
        train = train[:-needed]
        val.extend(moved)

    return train, val


# ======================================================
# Validation Check
# ======================================================

def assert_structural_coverage(
    train_set: List[Dict],
    full_universe: Dict[str, Set[str]],
) -> None:
    """
    Verify that the training split covers the complete structural
    feature universe of the dataset.

    Parameters
    ----------
    train_set : List[Dict]
        Training dataset.

    full_universe : Dict[str, Set[str]]
        Structural feature universe extracted from the full dataset.

    Raises
    ------
    ValueError
        Raised when one or more structural features are missing from
        the training split.
    """
    train_universe = build_feature_universe(train_set)

    for key in full_universe:
        missing = full_universe[key] - train_universe[key]
        if missing:
            raise ValueError(
                f"Training set missing {key}: {missing}"
            )

    print("Structural coverage verified")


# ======================================================
# IO Utilities
# ======================================================

def load_jsonl(path: str) -> List[Dict]:
    """
    Load a JSONL dataset into memory.

    Parameters
    ----------
    path : str
        Path to the JSONL file.

    Returns
    -------
    List[Dict]
        Parsed dataset records.
    """
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_jsonl(path: str, data: List[Dict]) -> None:
    """
    Save dataset records to a JSONL file.

    Parameters
    ----------
    path : str
        Output file path.

    data : List[Dict]
        Dataset records to persist.
    """
    with open(path, "w", encoding="utf-8") as f:
        for row in data:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")


# ======================================================
# Main
# ======================================================

def main() -> None:
    """
    Execute the complete structural dataset splitting workflow.

    Workflow
    --------
    1. Load augmented dataset
    2. Compute structural split
    3. Verify structural coverage
    4. Save train split
    5. Save validation split

    Returns
    -------
    None
    """
    input_path = Path(AUGMENTED_DATASETS_DIR) / "questions_paraphrased.jsonl"
    output_train = Path(SPLITS_DATASETS_DIR) / "train_base.jsonl"
    output_val = Path(SPLITS_DATASETS_DIR) / "val_base.jsonl"

    output_train.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_jsonl(input_path)
    print(f"Loaded {len(dataset)} samples")

    train, val = structural_split(dataset, val_ratio=0.2)

    universe = build_feature_universe(dataset)
    assert_structural_coverage(train, universe)

    save_jsonl(output_train, train)
    save_jsonl(output_val, val)

    print(f"Train size: {len(train)}")
    print(f"Val size:   {len(val)}")
    print(f"Split saved to {SPLITS_DATASETS_DIR}")


if __name__ == "__main__":
    main()
