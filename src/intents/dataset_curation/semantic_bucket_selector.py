"""
Semantic Bucket Selection
=========================

Semantic diversity control mechanism for synthetic intent datasets.

This module groups intents into semantic buckets based on structural
characteristics and ensures balanced representation across different
intent patterns.

Purpose
-------
Prevent dataset collapse into overly frequent patterns, such as:

- shallow paths dominating deeper ones
- filter-heavy intents dominating simple retrieval intents
- projection-heavy intents dominating minimal ones

By enforcing minimum coverage across semantic buckets, the resulting
dataset becomes more robust for:

- LLM fine-tuning
- query generalization
- compositional reasoning training

Design principles
-----------------
- lightweight semantic signature
- deterministic structure
- configurable diversity thresholds
- pipeline-compatible filtering stage

Dependencies
------------
collections.defaultdict
random
"""

from collections import defaultdict
import random


class SemanticBucketizer:
    """
    Builds compact semantic signatures for intents.

    The signature abstracts structural properties of the intent,
    allowing grouping by semantic complexity rather than exact schema.

    Signature components
    --------------------
    - primary intent type
    - normalized path depth
    - presence of filters
    - presence of aggregates
    - ordering usage
    - limit usage
    - projection size category
    """

    def bucket(self, intent):
        """
        Generate semantic bucket signature.

        Parameters
        ----------
        intent : Intent
            Structured intent object.

        Returns
        -------
        tuple
            Hashable semantic signature representing intent structure.
        """

        signature = []

        # primary intent category (e.g. retrieve, count, aggregate)
        signature.append(intent.intent.type)

        # normalize path depth to avoid explosion of bucket combinations
        # depth > 2 is treated uniformly
        depth = min(len(intent.schema_spec.path), 2)
        signature.append(f"depth_{depth}")

        # structural modifiers
        if intent.schema_spec.filters:
            signature.append("filter")

        if intent.schema_spec.aggregate:
            signature.append("aggregate")

        if intent.schema_spec.order_by:
            signature.append("order_by")

        if intent.schema_spec.limit:
            signature.append("limit")

        # normalize projection size to prevent sparsity
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
    Selects intents while preserving semantic diversity.

    Applies two-stage balancing strategy:

    Step 1
    ------
    Ensure minimum representation per semantic bucket.

    Step 2
    ------
    Optionally cap bucket size to prevent dominance of
    highly frequent structural patterns.

    Configuration parameters
    ------------------------
    min_per_bucket : int
        Minimum number of examples per semantic bucket.

    max_per_bucket : int or None
        Maximum number of examples per bucket.

    enable_semantic_balance : bool
        Toggle balancing behavior.
    """

    def __init__(self, config):
        """
        Initialize selector.

        Parameters
        ----------
        config : object
            Configuration object containing balancing parameters.
        """

        self.config = config

        self.bucketizer = SemanticBucketizer()

        self.min_per_bucket = getattr(config, "min_per_bucket", 20)

        self.max_per_bucket = getattr(config, "max_per_bucket", None)


    def select(self, intents):
        """
        Apply semantic diversity filtering to intent list.

        Parameters
        ----------
        intents : list
            List of structured intents.

        Returns
        -------
        list
            Balanced list of intents.
        """

        # bypass if balancing disabled
        if not getattr(self.config, "enable_semantic_balance", False):
            return intents

        grouped = defaultdict(list)

        # group intents by semantic signature
        for intent in intents:

            bucket = self.bucketizer.bucket(intent)

            grouped[bucket].append(intent)

        print(f"Semantic buckets discovered: {len(grouped)}")

        # shuffle examples within each bucket
        # avoids structural bias during selection
        for items in grouped.values():

            random.shuffle(items)

        dataset = []

        # ------------------------
        # step 1 — ensure minimum diversity
        # ------------------------

        for bucket, items in grouped.items():

            take = min(len(items), self.min_per_bucket)

            dataset.extend(items[:take])

            # store remaining examples for optional inclusion
            grouped[bucket] = items[take:]

        # ------------------------
        # step 2 — optionally expand dataset
        # ------------------------

        for bucket, items in grouped.items():

            if self.max_per_bucket is None:

                dataset.extend(items)

            else:

                # count current representation in dataset
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
