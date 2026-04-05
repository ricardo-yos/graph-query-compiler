"""
Combinatorial Structural Intent Generator
=========================================

Exhaustive combinatorial generator for graph query intent structures.

This module explores the space of possible query structures derived
from a graph schema by systematically combining:

- traversal paths
- projections (return attributes)
- filters
- aggregations
- ordering
- limits

The generator produces structurally valid intent specifications that
can later be converted into natural language questions or queries.

Design Goals
------------
- maximize structural coverage of the schema
- produce diverse query shapes
- enforce structural validity constraints
- support controlled combinatorial explosion via configuration
- enable reproducible dataset generation

Output
------
List[IntentSpec]

Each IntentSpec encodes a structured representation of a graph query,
including traversal path, constraints and structural modifiers.

Notes
-----
This module does NOT generate natural language.
It only produces structural representations.
"""

import itertools
import random
from typing import List

from .policies.numeric_policy import NUMERIC_ATTRIBUTES

from .intent_models import (
    IntentSpec,
    IntentCore,
    SchemaSpec,
    PrimaryIntent,
    AttributeFilter,
    AggregateSpec,
    StructuralModifier,
)

from .utils.path_utils import (
    get_allowed_targets,
    get_max_depth,
    allow_cycles,
)

from .utils.attribute_utils import (
    get_operators,
    get_aggregate_functions,
    get_filterable_attributes,
    sample_filter_value,
    enforce_mandatory_filters,
    get_orderable_attributes,
    is_aggregatable,
    get_returnable_attributes,
)


class CombinatorialStructuralGenerator:
    """
    Generate structurally valid graph query intents using controlled
    combinatorial expansion over schema components.

    Expansion dimensions include:

    - traversal depth
    - attribute projections
    - filter combinations
    - aggregation functions
    - ordering strategies
    - result limits

    Configuration controls combinatorial growth and structural constraints.
    """

    def __init__(self, schema, config):
        self.schema = schema
        self.config = config

    # ==================================================
    # PUBLIC API
    # ==================================================

    def generate(self) -> List[IntentSpec]:
        """
        Generate all structural intents under the configured regime.

        The generation process iterates over increasing traversal depth
        and expands each path into multiple structural variants.

        Returns
        -------
        List[IntentSpec]
            List of structurally valid intent specifications.
        """

        intents = []

        for depth in range(self.config.max_depth + 1):

            paths = self._generate_paths(depth)

            for path in paths:
                intents.extend(self._expand_path(path))

        # remove duplicated structures generated via different expansion paths
        if self.config.deduplicate_structures:
            intents = self._deduplicate(intents)

        return intents

    # ==================================================
    # PATH GENERATION
    # ==================================================

    def _generate_paths(self, depth):
        """
        Generate traversal paths up to a specified depth.

        Paths are constructed using DFS over outgoing relationships
        while respecting schema constraints such as:

        - allowed target labels
        - maximum traversal depth
        - cycle restrictions

        Parameters
        ----------
        depth : int
            Maximum traversal depth.

        Returns
        -------
        List
            List of raw path representations.
        """

        paths = []

        for label in self.schema.labels:

            # depth 0 represents direct queries on a single node
            if depth == 0:
                paths.append([label])
                continue

            max_depth = get_max_depth(label, depth)

            def dfs(current_path, current_label, remaining_depth):

                # any path with at least one edge is considered valid
                if len(current_path) > 1:
                    paths.append(current_path)

                if remaining_depth == 0:
                    return

                allowed_targets = get_allowed_targets(current_label)

                for rel in self.schema.get_outgoing(current_label):

                    # enforce schema-level traversal constraints
                    if allowed_targets and rel.target not in allowed_targets:
                        continue

                    # prevent cycles when disabled in regime configuration
                    if not allow_cycles(current_label):

                        visited = [
                            p if isinstance(p, str) else p[1]
                            for p in current_path
                        ]

                        if rel.target in visited:
                            continue

                    dfs(
                        current_path + [(rel.type, rel.target)],
                        rel.target,
                        remaining_depth - 1,
                    )

            dfs([label], label, max_depth)

        return paths

    # ==================================================
    # PATH EXPANSION
    # ==================================================

    def _expand_path(self, path):
        """
        Expand a traversal path into multiple structural query variants.

        Each path can produce multiple intents via combinations of:

        - projections
        - filters
        - aggregations
        - ordering
        - limits
        """

        intents = []

        target_candidates = self._get_target_candidates(path)

        for target_label in target_candidates:

            attributes = get_returnable_attributes(target_label)

            # skip nodes that cannot produce meaningful outputs
            if not attributes:
                continue

            projections = self._expand_projections(target_label)

            for projection in projections:

                base_intent = IntentSpec(
                    intent=IntentCore(
                        type=PrimaryIntent.RETRIEVE,
                        modifiers=[],
                    ),
                    schema=SchemaSpec(
                        target={"label": target_label},

                        path=self._build_path_spec(path),

                        filters=[],

                        order_by=None,

                        limit=None,

                        aggregate=None,

                        return_attributes=list(projection),
                    ),
                )

                expanded = [base_intent]

                expanded = self._expand_filters_variants(expanded)

                if self.config.allow_aggregation:
                    expanded = self._expand_aggregation(expanded)

                if self.config.allow_order_by:
                    expanded = self._expand_order_by(expanded)

                if self.config.allow_limit:
                    expanded = self._expand_limit(expanded)

                intents.extend(expanded)

        return intents

    # ==================================================
    # PROJECTION
    # ==================================================

    def _expand_projections(self, label, aggregate=None):
        """
        Determine attribute projection combinations.

        Delegates selection logic to attribute utilities, allowing
        centralized control of projection policies.
        """

        return get_returnable_attributes(label, aggregate)

    # ==================================================
    # FILTERS EXPANSION
    # ==================================================

    def _expand_filters_variants(self, intents):
        """
        Expand intents with all valid filter combinations defined
        by the regime configuration.

        Supports:
        - single attribute filters
        - multi-attribute filters
        - operator constraints
        - compatibility validation
        """

        all_variants = []

        for intent in intents:
            all_variants.extend(
                self._expand_filters_single(intent)
            )

        return all_variants


    def _expand_filters_single(self, intent: IntentSpec):
        """
        Generate filter combinations for a single intent.

        Ensures:
        - attribute/operator compatibility
        - no duplicated attributes in same combination
        - numeric range consistency
        """

        variants = []

        labels = self._get_all_labels(intent)

        filter_candidates_by_label = {}

        # build candidate filters per node label
        for label in labels:

            attributes = get_filterable_attributes(label)

            filter_candidates = []

            for attr in attributes:

                operators = get_operators(label, attr)

                if not operators:
                    continue

                # limit combinatorial explosion via operator sampling
                k = min(
                    len(operators),
                    self.config.max_operators_per_attribute
                )

                sampled_ops = random.sample(operators, k)

                for op in sampled_ops:

                    filter_candidates.append(
                        AttributeFilter(
                            node_label=label,
                            attribute=attr,
                            operator=op,
                            value=sample_filter_value(attr, label),
                        )
                    )

            filter_candidates_by_label[label] = filter_candidates

        # ------------------------
        # single filter variants
        # ------------------------
        for label, candidates in filter_candidates_by_label.items():

            for f in candidates:

                new_intent = intent.model_copy(deep=True)

                filters = enforce_mandatory_filters(label, [f])

                new_intent.schema_spec.filters.extend(filters)

                self._add_modifier(
                    new_intent,
                    StructuralModifier.FILTER
                )

                variants.append(new_intent)

        # ------------------------
        # multi-filter variants
        # ------------------------
        if self.config.allow_multiple_filters:

            for label, candidates in filter_candidates_by_label.items():

                max_k = min(
                    self.config.max_filters_per_node,
                    len(candidates)
                )

                for k in range(2, max_k + 1):

                    for combo in itertools.combinations(candidates, k):

                        # prevent duplicate attribute constraints
                        attrs = {f.attribute for f in combo}

                        if len(attrs) != len(combo):
                            continue

                        # ensure numeric filters produce valid ranges
                        if not self._filters_are_compatible(combo):
                            continue

                        new_intent = intent.model_copy(deep=True)

                        filters = enforce_mandatory_filters(
                            label,
                            list(combo),
                        )

                        new_intent.schema_spec.filters.extend(filters)

                        self._add_modifier(
                            new_intent,
                            StructuralModifier.FILTER
                        )

                        variants.append(new_intent)

        return variants


    def _filters_are_compatible(self, filters):
        """
        Ensure numeric filter combinations define valid intervals.

        Example:
        price >= 10 AND price <= 5 -> invalid
        """

        grouped = {}

        for f in filters:
            grouped.setdefault(f.attribute, []).append(f)

        for attr, flist in grouped.items():

            if attr not in NUMERIC_ATTRIBUTES:
                continue

            min_val = None
            max_val = None

            for f in flist:

                if f.operator in (">", ">="):

                    min_val = (
                        f.value
                        if min_val is None
                        else max(min_val, f.value)
                    )

                elif f.operator in ("<", "<="):

                    max_val = (
                        f.value
                        if max_val is None
                        else min(max_val, f.value)
                    )

            if min_val is not None and max_val is not None:

                if min_val > max_val:
                    return False

        return True

    # ==================================================
    # AGGREGATION
    # ==================================================

    def _expand_aggregation(self, intents):
        """
        Generate aggregation variants including:

        - attribute aggregations (avg, min, max, sum)
        - COUNT(*)
        """

        expanded = []

        for intent in intents:

            attributes = intent.schema_spec.return_attributes

            for attr in attributes:

                if not is_aggregatable(attr):
                    continue

                for fn in get_aggregate_functions(attr):

                    new_intent = intent.model_copy(deep=True)

                    new_intent.schema_spec.aggregate = AggregateSpec(
                        function=fn,
                        attribute=attr,
                    )

                    # aggregate results replace attribute projections
                    new_intent.schema_spec.return_attributes = []

                    self._add_modifier(
                        new_intent,
                        StructuralModifier.AGGREGATE
                    )

                    expanded.append(new_intent)

            # COUNT(*) independent of attributes
            for fn in get_aggregate_functions(None):

                new_intent = intent.model_copy(deep=True)

                new_intent.schema_spec.aggregate = AggregateSpec(
                    function=fn,
                    attribute=None,
                )

                new_intent.schema_spec.return_attributes = []

                self._add_modifier(
                    new_intent,
                    StructuralModifier.AGGREGATE
                )

                expanded.append(new_intent)

        return expanded

    # ==================================================
    # ORDER BY
    # ==================================================

    def _expand_order_by(self, intents):
        """
        Generate ranking variants based on orderable attributes.

        Aggregated intents can only be ordered by the aggregated attribute.
        """

        expanded = []

        for intent in intents:

            label = intent.schema_spec.target["label"]

            if intent.schema_spec.aggregate:

                agg_attr = intent.schema_spec.aggregate.attribute

                if agg_attr is None:
                    continue

                attributes = [agg_attr]

            else:

                attributes = get_orderable_attributes(label)

            for attr in attributes:

                for direction in self.config.order_by_directions:

                    new_intent = intent.model_copy(deep=True)

                    new_intent.schema_spec.order_by = {
                        "attribute": attr,
                        "direction": direction,
                    }

                    self._add_modifier(
                        new_intent,
                        StructuralModifier.ORDER_BY
                    )

                    expanded.append(new_intent)

        return expanded

    # ==================================================
    # LIMIT
    # ==================================================

    def _expand_limit(self, intents):
        """
        Generate variants with result size constraints.
        """

        expanded = []

        for intent in intents:

            for value in self.config.limit_values:

                new_intent = intent.model_copy(deep=True)

                new_intent.schema_spec.limit = value

                self._add_modifier(
                    new_intent,
                    StructuralModifier.LIMIT
                )

                expanded.append(new_intent)

        return expanded

    # ==================================================
    # UTILITIES
    # ==================================================

    def _add_modifier(self, intent, modifier):
        """
        Register structural modifier if not already present.
        """

        if modifier not in intent.intent.modifiers:
            intent.intent.modifiers.append(modifier)

    def _get_target_candidates(self, path):
        """
        Determine valid target node from traversal path.
        """

        last_step = path[-1]

        if isinstance(last_step, str):
            return [last_step]

        return [last_step[1]]

    def _build_path_spec(self, raw_path):
        """
        Convert internal path representation into schema format.
        """

        path_spec = []

        for i in range(len(raw_path) - 1):

            source = raw_path[i]

            if isinstance(raw_path[i + 1], tuple):

                rel_type, target = raw_path[i + 1]

                path_spec.append(
                    {
                        "source": {"label": source},
                        "relationship": rel_type,
                        "target": {"label": target},
                    }
                )

        return path_spec

    def _get_all_labels(self, intent):
        """
        Collect all node labels involved in the query structure.
        """

        labels = set()

        labels.add(intent.schema_spec.target["label"])

        for step in intent.schema_spec.path:

            labels.add(step["source"]["label"])
            labels.add(step["target"]["label"])

        return list(labels)

    def _deduplicate(self, intents):
        """
        Remove structurally identical intents.

        Uses serialized representation as structural fingerprint.
        """

        seen = set()

        unique = []

        for intent in intents:

            key = intent.model_dump_json()

            if key not in seen:

                seen.add(key)

                unique.append(intent)

        return unique
