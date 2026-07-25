"""
Benchmark Evaluation Module
===========================

Provides evaluation utilities for measuring the GQC model
performance on structured intent generation tasks.

Metrics
-------
- regime accuracy
- exact schema match
- field-level accuracy
- component-level accuracy
- schema mismatch analysis

The evaluator compares model predictions against gold
schemas while handling order-independent structures such as
filters and return attributes.
"""


import json
from collections import defaultdict


class BenchmarkEvaluator:
    """
    Evaluate structured query intent predictions.

    Computes regime classification accuracy and schema
    generation metrics by comparing predicted outputs against
    benchmark ground truth examples.
    """

    def __init__(self):
        """
        Initialize evaluation counters and statistics.
        """

        self.stats = {
            "total": 0,
            "regime_correct": 0,
            "exact_schema_match": 0,
        }

        self.field_stats = {
            "correct": 0,
            "total": 0,
        }

        self.error_counter = defaultdict(int)

        self.component_stats = defaultdict(
            lambda: {
                "correct": 0,
                "total": 0,
            }
        )


    def _normalize(self, obj):
        """
        Normalize schema structures before comparison.

        Makes unordered structures comparable by sorting:
        - lists of dictionaries
        - lists of primitive values

        This avoids false mismatches caused by ordering
        differences in schema fields.
        """

        if isinstance(obj, dict):

            return {
                key: self._normalize(value)
                for key, value in obj.items()
            }


        if isinstance(obj, list):

            if not obj:
                return []


            if isinstance(obj[0], dict):

                normalized = [
                    self._normalize(item)
                    for item in obj
                ]

                return sorted(
                    normalized,
                    key=lambda x: json.dumps(
                        x,
                        sort_keys=True,
                    )
                )


            return sorted(obj)


        return obj


    def load_jsonl(self, file_path):
        """
        Load benchmark examples from a JSONL file.

        Parameters
        ----------
        file_path : str | Path
            Benchmark dataset path.

        Returns
        -------
        list
            Loaded benchmark examples.
        """

        examples = []

        with open(
            file_path,
            "r",
            encoding="utf-8",
        ) as file:

            for line in file:
                examples.append(
                    json.loads(line)
                )

        return examples


    def compare(
        self,
        gold,
        pred,
        path="",
    ):
        """
        Recursively compare expected and predicted schemas.

        Returns detailed mismatch information including:
        - field path
        - expected value
        - predicted value

        Parameters
        ----------
        gold : Any
            Expected schema.

        pred : Any
            Predicted schema.

        path : str
            Current schema location.

        Returns
        -------
        list
            Schema differences.
        """

        errors = []

        if type(gold) != type(pred):

            return [{
                "path": path,
                "expected": gold,
                "predicted": pred,
            }]


        if isinstance(gold, dict):

            keys = set(gold.keys()) | set(pred.keys())

            for key in keys:

                new_path = (
                    f"{path}.{key}"
                    if path
                    else key
                )

                if key not in gold:

                    errors.append({
                        "path": new_path,
                        "expected": "<missing>",
                        "predicted": pred[key],
                    })

                    continue


                if key not in pred:

                    errors.append({
                        "path": new_path,
                        "expected": gold[key],
                        "predicted": "<missing>",
                    })

                    continue


                errors.extend(
                    self.compare(
                        gold[key],
                        pred[key],
                        new_path,
                    )
                )


        elif isinstance(gold, list):

            if len(gold) != len(pred):

                errors.append({
                    "path": path,
                    "expected": len(gold),
                    "predicted": len(pred),
                })


            for index in range(
                min(len(gold), len(pred))
            ):

                errors.extend(
                    self.compare(
                        gold[index],
                        pred[index],
                        f"{path}[{index}]",
                    )
                )


        elif gold != pred:

            errors.append({
                "path": path,
                "expected": gold,
                "predicted": pred,
            })


        return errors


    def evaluate_example(
        self,
        example,
        predicted_regime,
        predicted_schema,
    ):
        """
        Evaluate a single benchmark example.

        Updates global statistics and returns the
        per-example evaluation result.
        """

        self.stats["total"] += 1

        gold_schema = self._normalize(
            example["schema"]
        )

        pred_schema = self._normalize(
            predicted_schema
        )


        result = {
            "question": example["question"],
            "expected_regime": example["regime"],
            "predicted_regime": predicted_regime,
            "regime_correct": False,
            "exact_schema_match": False,
            "errors": [],
        }


        if predicted_regime == example["regime"]:

            self.stats["regime_correct"] += 1
            result["regime_correct"] = True


        self._update_component_stats(
            gold_schema,
            pred_schema,
        )

        self._update_field_stats(
            gold_schema,
            pred_schema,
        )


        schema_errors = self.compare(
            gold_schema,
            pred_schema,
        )


        if not schema_errors:

            self.stats["exact_schema_match"] += 1
            result["exact_schema_match"] = True


        else:

            for error in schema_errors:
                self.error_counter[
                    error["path"]
                ] += 1


            result["errors"] = schema_errors
            result["expected_schema"] = example["schema"]
            result["predicted_schema"] = predicted_schema


        return result
