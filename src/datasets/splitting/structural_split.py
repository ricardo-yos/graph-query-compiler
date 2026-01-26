"""
Structural Dataset Split for Graph Reasoning
============================================

Structure-aware train/validation split for datasets in which each
example represents a structured graph query schema.

This split strategy guarantees:
- Full structural coverage in the training set
- Validation set composed only of recombinations of known structures
- No leakage of unseen graph primitives into validation

Input
-----
BASE_DATASETS_DIR / questions_base.jsonl

Output
------
SPLITS_DATASETS_DIR /
    ├── train_base.jsonl
    └── val_base.jsonl
"""

import json
import random
from typing import Dict, List, Set, Tuple

from pathlib import Path
from config.paths import BASE_DATASETS_DIR, SPLITS_DATASETS_DIR


# ======================================================
# Feature Extraction
# ======================================================

def extract_features(schema: Dict) -> Dict[str, Set[str]]:
    """
    Extract structural features from a graph query schema.

    Parameters
    ----------
    schema : dict
        Structured graph query schema.

    Returns
    -------
    Dict[str, Set[str]]
        Dictionary containing sets of:
        - labels
        - relations
        - attributes
        - operators
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

    Parameters
    ----------
    dataset : List[dict]
        Dataset containing question-schema pairs.

    Returns
    -------
    Dict[str, Set[str]]
        Aggregated universe of all structural features.
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

    The training set is guaranteed to cover the full
    structural feature universe of the dataset.

    Parameters
    ----------
    dataset : List[dict]
        Input dataset.
    val_ratio : float, optional
        Proportion of samples assigned to validation.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Tuple[List[dict], List[dict]]
        Training and validation splits.
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
        Check whether a sample introduces unseen structural features.
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
    Assert that the training set covers all structural features.

    Parameters
    ----------
    train_set : List[dict]
        Training dataset split.
    full_universe : Dict[str, Set[str]]
        Structural universe derived from the full dataset.

    Raises
    ------
    ValueError
        If any structural feature is missing from the training set.
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
    Load a JSONL file into memory.
    """
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_jsonl(path: str, data: List[Dict]) -> None:
    """
    Save a list of dictionaries to a JSONL file.
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
    Execute the structure-aware dataset split.
    """
    input_path = Path(BASE_DATASETS_DIR) / "questions_base.jsonl"
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
