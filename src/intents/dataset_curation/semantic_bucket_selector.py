"""
Semantic Bucket Selection
=========================

Semantic diversity balancing utilities for synthetic structural
intent datasets.

This module groups structurally similar intents into semantic buckets
and applies configurable balancing strategies to improve dataset
diversity and structural coverage.

Purpose
-------
Reduce structural imbalance caused by highly frequent query patterns,
such as:

- shallow traversal paths dominating deeper traversals
- filter-heavy queries dominating simple retrieval intents
- projection-heavy structures dominating minimal projections
- repetitive structural regimes overwhelming rarer compositions

Balanced semantic coverage improves robustness for:

- LLM fine-tuning
- semantic parsing
- query generalization
- compositional reasoning tasks

Design Principles
-----------------
- lightweight structural signatures
- deterministic semantic grouping
- configurable diversity balancing
- pipeline-compatible dataset filtering
- controlled preservation of rare structures

Dependencies
------------
collections.defaultdict
random
"""

from collections import defaultdict
import random


class SemanticBucketizer:
    """
    Build compact semantic signatures for structural query intents.

    Signatures abstract structural characteristics of an intent,
    enabling grouping by semantic complexity rather than exact
    schema representation.

    Signature Components
    --------------------
    - structural regime
    - normalized traversal depth
    - filter presence
    - aggregation presence
    - ordering presence
    - limit usage
    - normalized projection size
    """

    def bucket(self, intent):
        """
        Generate semantic bucket signature for a structural intent.

        Parameters
        ----------
        intent : IntentSpec
            Structured query intent object.

        Returns
        -------
        tuple
            Hashable semantic signature representing structural
            query characteristics.
        """

        signature = []

        # structural regime identifier
        signature.append(
            intent.intent.regime or "unknown"
        )

        # normalize traversal depth to reduce bucket sparsity
        # all depths greater than 2 are grouped together
        depth = min(len(intent.schema_spec.path), 2)
        signature.append(f"depth_{depth}")

        # structural modifier presence indicators
        if intent.schema_spec.filters:
            signature.append("filter")

        if intent.schema_spec.aggregate:
            signature.append("aggregate")

        if intent.schema_spec.order_by:
            signature.append("order_by")

        if intent.schema_spec.limit:
            signature.append("limit")

        # normalize projection size to prevent excessive
        # fragmentation of semantic buckets
        projections = len(intent.schema_spec.return_attributes or [])

        if projections == 0:
            signature.append("proj_0")

        elif projections == 1:
            signature.append("proj_1")

        elif projections == 2:
            signature.append("proj_2")

        else:
            signature.append("proj_many")

        return tuple(signature)


class SemanticBucketSelector:
    """
    Select structural intents while preserving semantic diversity.

    Applies a two-stage semantic balancing strategy:

    Step 1
    ------
    Ensure minimum representation across semantic buckets.

    Step 2
    ------
    Optionally cap bucket sizes to prevent highly frequent
    structural patterns from dominating the dataset.

    Configuration Parameters
    ------------------------
    min_per_bucket : int
        Minimum number of examples preserved per semantic bucket.

    max_per_bucket : int or None
        Maximum number of examples allowed per bucket.

    enable_semantic_balance : bool
        Enable or disable semantic balancing behavior.
    """

    def __init__(self, config):
        """
        Initialize semantic diversity selector.

        Parameters
        ----------
        config : object
            Configuration object containing semantic balancing
            parameters and selection policies.
        """

        self.config = config

        self.bucketizer = SemanticBucketizer()

        self.min_per_bucket = getattr(config, "min_per_bucket", 20)

        self.max_per_bucket = getattr(config, "max_per_bucket", None)


    def select(self, intents):
        """
        Apply semantic diversity balancing to structural intents.

        The balancing process:

        1. groups intents into semantic buckets
        2. guarantees minimum bucket representation
        3. optionally limits dominant bucket sizes

        Parameters
        ----------
        intents : list
            List of structural intent objects.

        Returns
        -------
        list
            Semantically balanced structural intent dataset.
        """

        # bypass balancing when semantic diversity control is disabled
        if not getattr(self.config, "enable_semantic_balance", False):
            return intents

        grouped = defaultdict(list)

        # group intents by semantic structural signature
        for intent in intents:

            bucket = self.bucketizer.bucket(intent)

            grouped[bucket].append(intent)

        print(f"Semantic buckets discovered: {len(grouped)}")

        # shuffle bucket contents to reduce ordering bias
        # during dataset selection
        for items in grouped.values():

            random.shuffle(items)

        dataset = []

        # ------------------------
        # # step 1 — guarantee minimum semantic coverage
        # ------------------------

        for bucket, items in grouped.items():

            take = min(len(items), self.min_per_bucket)

            dataset.extend(items[:take])

            # preserve remaining examples for optional expansion
            grouped[bucket] = items[take:]

        # ------------------------
        # step 2 — optionally expand dataset while controlling
        # dominance of highly frequent structural patterns
        # ------------------------

        for bucket, items in grouped.items():

            if self.max_per_bucket is None:

                dataset.extend(items)

            else:

                # compute current bucket representation in dataset
                current = sum(
                    1
                    for x in dataset
                    if self.bucketizer.bucket(x) == bucket
                )

                remaining = max(
                    0,
                    self.max_per_bucket - current
                )

                dataset.extend(items[:remaining])

        return dataset
