"""
Dataset Structural Coverage Reports
==================================

Generates diagnostic reports describing the structural coverage
of a previously generated intent dataset.

The goal of this module is to verify whether the generated dataset
adequately explores the structural space defined by the graph schema.

This helps ensure:

- good coverage of graph paths
- balanced structural diversity
- sufficient representation of query patterns
- early detection of structural biases

Input:
    structural_intents.jsonl

Output:
    path_coverage_report.json

Typical usage:
    run after dataset generation to validate structural diversity
    before training or fine-tuning models.

Reports currently included:
    1. Path coverage report
       Measures how graph paths are distributed across the dataset.

Future reports may include:
    - attribute coverage
    - operator distribution
    - regime distribution
    - filter complexity distribution
"""

import json

from config.paths import INTENTS_DATA_DIR

from src.intents.reports.path_coverage_report import path_coverage_report
from src.intents.reports.save_report import save_report


def load_dataset():
    """
    Loads the generated intent dataset from JSONL format.

    Each line in the file represents a single intent serialized as JSON.

    Returns
    -------
    list[dict]
        List of intent dictionaries ready for structural analysis.

    Raises
    ------
    FileNotFoundError
        If the dataset file does not exist.
    """

    dataset_path = INTENTS_DATA_DIR / "structural_intents.jsonl"

    if not dataset_path.exists():

        raise FileNotFoundError(

            f"Dataset not found: {dataset_path}"

        )

    intents = []

    # JSONL format: one JSON object per line
    with open(dataset_path, encoding="utf-8") as f:

        for line in f:

            intents.append(json.loads(line))

    return intents


def main():
    """
    Main execution pipeline for dataset diagnostics.

    Steps
    -----
    1. Load dataset
    2. Compute structural reports
    3. Print results to console
    4. Save reports to disk

    Designed to be easily extended with additional diagnostics.
    """

    # --------------------------------------------------
    # Load dataset
    # --------------------------------------------------

    intents = load_dataset()

    print("\n" + "=" * 50)
    print("DATASET REPORTS")
    print("=" * 50)

    # --------------------------------------------------
    # 1. Path coverage
    # --------------------------------------------------
    # Measures how frequently each graph path appears.
    # Useful for detecting structural bias in generated intents.

    print("\n1) Path Coverage Report")
    print("-" * 50)

    path_reports = path_coverage_report(intents)

    # Print each structural metric returned by the report
    for name, report in path_reports.items():

        print(f"\n{name}")
        print(report)

    # --------------------------------------------------
    # Save report
    # --------------------------------------------------

    save_report(
        path_reports,
        "path_coverage_report.json"
    )


if __name__ == "__main__":

    main()
