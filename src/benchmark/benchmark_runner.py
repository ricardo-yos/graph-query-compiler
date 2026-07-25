"""
Benchmark Runner
================

Execute structured intent generation benchmarks for the GQC model.

Workflow
--------
1. Load benchmark examples
2. Generate predictions using the inference pipeline
3. Compare predictions against expected schemas
4. Aggregate evaluation metrics
5. Rank the most frequent schema errors
6. Save benchmark reports

Output
------
Generates a JSON report containing:

- global benchmark metrics
- per-example evaluation results
- ranked schema mismatch errors
"""


import json
from pathlib import Path
from collections import Counter

from .benchmark_evaluator import BenchmarkEvaluator
from src.inference.inference_llamacpp import predict

from config.paths import (
    BENCHMARK_DATA_DIR,
    BENCHMARK_REPORTS_DIR,
)


# ============================================================
# Benchmark configuration
# ============================================================

# Benchmark dataset identifier.
BENCHMARK_NAME = "benchmark_simple_regimes_v1"


# Input benchmark file containing questions and expected schemas.
BENCHMARK_FILE = (
    Path(BENCHMARK_DATA_DIR)
    / f"{BENCHMARK_NAME}.jsonl"
)


# Output file containing evaluation results.
RESULTS_FILE = (
    Path(BENCHMARK_REPORTS_DIR)
    / f"{BENCHMARK_NAME}_results.json"
)


# ============================================================
# Error analysis
# ============================================================

def normalize_error_value(value):
    """
    Convert error values into comparable string representations.

    Complex structures such as dictionaries and lists are
    serialized to ensure they can be counted during aggregation.

    Parameters
    ----------
    value : Any
        Error value.

    Returns
    -------
    str
        Normalized string representation.
    """

    if isinstance(value, (dict, list)):
        return json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
        )

    return str(value)


def rank_schema_errors(results: list) -> list:
    """
    Aggregate and rank the most frequent schema mismatches.

    Groups errors by:
    - schema path
    - expected value
    - predicted value

    Parameters
    ----------
    results : list
        Per-example benchmark evaluation results.

    Returns
    -------
    list
        Ranked schema error statistics.
    """

    error_counter = Counter()

    for result in results:

        errors = result.get(
            "errors",
            []
        )

        for error in errors:

            key = (
                str(error["path"]),
                normalize_error_value(
                    error["expected"]
                ),
                normalize_error_value(
                    error["predicted"]
                ),
            )

            error_counter[key] += 1


    ranked_errors = []

    for (
        path,
        expected,
        predicted,
    ), count in error_counter.most_common():

        ranked_errors.append(
            {
                "path": path,
                "expected": expected,
                "predicted": predicted,
                "count": count,
            }
        )

    return ranked_errors


# ============================================================
# Benchmark execution
# ============================================================

def run_benchmark() -> dict:
    """
    Execute benchmark evaluation over all examples.

    The function runs model inference, evaluates predictions,
    aggregates metrics, analyzes schema errors, and saves
    the final benchmark report.

    Returns
    -------
    dict
        Global benchmark evaluation report.
    """

    evaluator = BenchmarkEvaluator()

    examples = evaluator.load_jsonl(
        BENCHMARK_FILE
    )

    results = []

    total_examples = len(examples)

    failed_examples = 0


    print("\n" + "=" * 60)
    print("BENCHMARK EXECUTION")
    print("=" * 60)

    print(
        f"Examples: {total_examples}"
    )


    for index, example in enumerate(
        examples,
        start=1,
    ):

        print(
            f"[{index}/{total_examples}] "
            f"{example['question']}"
        )


        try:

            # Generate structured prediction.
            prediction = predict(
                example["question"],
                debug=False,
            )


            predicted_regime = prediction.get(
                "regime"
            )

            predicted_schema = prediction.get(
                "schema"
            )


            # Compare prediction against expected schema.
            result = evaluator.evaluate_example(
                example=example,
                predicted_regime=predicted_regime,
                predicted_schema=predicted_schema,
            )


        except Exception as exception:

            failed_examples += 1

            print(
                f"Benchmark failure: {exception}"
            )


            result = {
                "index": index,
                "question": example["question"],
                "expected_regime": example.get("regime"),
                "predicted_regime": None,
                "predicted_schema": None,
                "correct_regime": False,
                "correct_schema": False,
                "error": str(exception),
                "errors": [],
            }


        results.append(result)


    # Generate aggregated metrics.
    report = evaluator.generate_report()


    # Add error analysis.
    report["errors"] = rank_schema_errors(
        results
    )


    report["total_schema_errors"] = sum(
        error["count"]
        for error in report["errors"]
    )


    report["unique_error_types"] = len(
        report["errors"]
    )


    report["total_examples"] = total_examples
    report["failed_examples"] = failed_examples


    save_results(
        results,
        report,
    )


    return report


# ============================================================
# Results persistence
# ============================================================

def save_results(
    results: list,
    report: dict,
) -> None:
    """
    Save benchmark results as JSON.

    Parameters
    ----------
    results : list
        Per-example evaluation results.

    report : dict
        Aggregated benchmark metrics.
    """

    output = {
        "report": report,
        "results": results,
    }


    with open(
        RESULTS_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=4,
            ensure_ascii=False,
        )


    print(
        f"Results saved to: {RESULTS_FILE}"
    )


# ============================================================
# Entry point
# ============================================================

def main() -> None:
    """
    Execute benchmark evaluation and print final report.
    """

    report = run_benchmark()

    print()
    print("=" * 60)
    print("FINAL REPORT")
    print("=" * 60)

    print(
        json.dumps(
            report,
            indent=4,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
