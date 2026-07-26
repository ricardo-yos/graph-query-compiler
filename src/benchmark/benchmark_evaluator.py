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

    # ============================================================
    # Schema normalization
    # ============================================================

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

    # ============================================================
    # Dataset loading
    # ============================================================

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

    # ============================================================
    # Schema comparison
    # ============================================================

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

    # ============================================================
    # Example evaluation
    # ============================================================

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

    # ============================================================
    # Field-level statistics
    # ============================================================

    def _count_leaf_fields(self, obj):
        """
        Count terminal values in a nested schema structure.

        Used to estimate the number of comparable fields when
        schemas have missing or incompatible structures.
        """

        if isinstance(obj, dict):
            return sum(
                self._count_leaf_fields(v)
                for v in obj.values()
            )

        if isinstance(obj, list):
            return sum(
                self._count_leaf_fields(x)
                for x in obj
            )

        return 1


    def _update_field_stats(self, gold, pred):
        """
        Update field-level accuracy statistics.

        Compares primitive schema values recursively and counts
        matching fields against the total number of evaluated
        fields.
        """

        if type(gold) != type(pred):

            self.field_stats["total"] += (
                self._count_leaf_fields(gold)
            )

            return


        if isinstance(gold, dict):

            all_keys = set(gold.keys()) | set(pred.keys())

            for key in all_keys:

                if key not in gold:

                    self.field_stats["total"] += (
                        self._count_leaf_fields(pred[key])
                    )

                    continue


                if key not in pred:

                    self.field_stats["total"] += (
                        self._count_leaf_fields(gold[key])
                    )

                    continue


                self._update_field_stats(
                    gold[key],
                    pred[key],
                )


        elif isinstance(gold, list):

            for i in range(
                min(len(gold), len(pred))
            ):

                self._update_field_stats(
                    gold[i],
                    pred[i],
                )


        else:

            self.field_stats["total"] += 1

            if gold == pred:
                self.field_stats["correct"] += 1

    # ============================================================
    # Component-level statistics
    # ============================================================

    def _update_component_stats(
        self,
        gold,
        pred,
        path="",
    ):
        """
        Update accuracy statistics for individual schema components.

        Tracks correctness for each schema path, allowing detailed
        error analysis such as filters, aggregations, ordering,
        and return attributes.
        """

        if type(gold) != type(pred):

            self.component_stats[path]["total"] += 1

            return


        if isinstance(gold, dict):

            all_keys = set(gold.keys()) | set(pred.keys())

            for key in all_keys:

                new_path = (
                    f"{path}.{key}"
                    if path
                    else key
                )


                if key not in gold or key not in pred:

                    self.component_stats[new_path]["total"] += 1

                    continue


                self._update_component_stats(
                    gold[key],
                    pred[key],
                    new_path,
                )


        elif isinstance(gold, list):

            if len(gold) != len(pred):

                self.component_stats[path]["total"] += 1


            for i in range(
                min(len(gold), len(pred))
            ):

                self._update_component_stats(
                    gold[i],
                    pred[i],
                    f"{path}[{i}]",
                )


        else:

            self.component_stats[path]["total"] += 1

            if gold == pred:
                self.component_stats[path]["correct"] += 1

    # ============================================================
    # Report generation
    # ============================================================

    def generate_report(self):
        """
        Generate aggregated benchmark evaluation metrics.

        Returns
        -------
        dict
            Benchmark report containing:
            - regime accuracy
            - exact schema match
            - field accuracy
            - component-level accuracy
            - most frequent schema errors
        """

        total = self.stats["total"]


        if total == 0:

            return {
                "total_examples": 0,
                "regime_accuracy": 0.0,
                "exact_schema_match": 0.0,
                "field_accuracy": 0.0,
                "component_accuracy": {},
                "most_common_errors": [],
            }


        field_accuracy = (
            round(
                self.field_stats["correct"]
                / self.field_stats["total"]
                * 100,
                2,
            )
            if self.field_stats["total"] > 0
            else 0.0
        )


        component_accuracy = {}

        for component, values in self.component_stats.items():

            if values["total"] == 0:
                continue


            component_accuracy[component] = round(
                values["correct"]
                / values["total"]
                * 100,
                2,
            )


        most_common_errors = [
            {
                "path": key,
                "count": value,
            }
            for key, value in sorted(
                self.error_counter.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:20]
        ]


        return {
            "total_examples": total,
            "regime_accuracy": round(
                self.stats["regime_correct"]
                / total
                * 100,
                2,
            ),
            "exact_schema_match": round(
                self.stats["exact_schema_match"]
                / total
                * 100,
                2,
            ),
            "field_accuracy": field_accuracy,
            "component_accuracy": dict(
                sorted(component_accuracy.items())
            ),
            "most_common_errors": most_common_errors,
        }
