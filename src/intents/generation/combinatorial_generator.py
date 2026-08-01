"""
Combinatorial Structural Intent Generator
=========================================

Controlled generator for graph query intent structures.

This module explores the structural search space of graph queries by
combining schema-driven components:

- traversal paths
- return projections
- filter constraints
- aggregation operations
- ordering strategies
- result limits

Generation is controlled through regime-specific policies that define
structural constraints, sampling strategies, and combinatorial limits.

The generated structures represent intermediate query intents that can
later be transformed into natural language questions or compiled into
executable graph queries.

Design Goals
------------

- maximize structural coverage of supported query patterns
- generate diverse but valid query structures
- enforce schema and regime constraints
- control combinatorial growth through configurable policies
- support reproducible dataset generation

Output
------

List[IntentSpec]

Each IntentSpec contains a structured representation of a graph query,
including traversal information, constraints, projections, and modifiers.

Notes
-----

This module does not generate natural language.
It only creates structured query representations.
"""

import itertools
import random
from typing import List

from .policies.numeric_policy import NUMERIC_ATTRIBUTES
from .policies.regime_policy import REGIME_POLICY
from .policies.filter_policy import FILTER_GENERATION_POLICY

from .intent_models import (
    IntentSpec,
    IntentCore,
    SchemaSpec,
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
    get_aggregatable_attributes,
    get_returnable_attributes,
)


class CombinatorialStructuralGenerator:
    """
    Generate structurally valid graph query intents through controlled
    combinatorial expansion over schema components.

    The generator explores combinations of:

    - traversal structures
    - attribute projections
    - filter constraints
    - aggregation operations
    - ordering strategies
    - result limits

    Expansion behavior is constrained by regime policies and configuration
    parameters to maintain structural diversity while preventing excessive
    combinatorial growth.
    """

    def __init__(self, schema, config):
        self.schema = schema
        self.config = config
        self.regime_policy = REGIME_POLICY[
            self.config.regime_name
        ]        

    # ==================================================
    # PUBLIC API
    # ==================================================

    def generate(self) -> List[IntentSpec]:
        """
        Generate a collection of structurally valid query intents.

        The generation process starts from schema traversal paths and expands
        each path into different structural configurations according to the
        active regime policy.

        Expansion includes structural dimensions such as projections, filters,
        aggregations, ordering and limits, depending on the enabled features.

        Returns
        -------
        List[IntentSpec]
            Generated query intent structures satisfying schema and regime
            constraints.
        """

        max_paths = self.regime_policy["max_paths"]
        
        intents = []

        start_depth = 0

        if self.config.regime_name.startswith(
            "relational"
        ):
            start_depth = 1

        for depth in range(
            start_depth,
            self.config.max_depth + 1
        ):

            paths = self._generate_paths(depth)

            if len(paths) > max_paths:

                paths = random.sample(
                    paths,
                    max_paths
                )

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
        Generate valid traversal paths up to a given depth.

        Paths are created through graph traversal while respecting schema
        constraints and regime-specific rules.

        The generation process considers:

        - allowed target nodes
        - maximum traversal depth
        - cycle restrictions

        Parameters
        ----------
        depth : int
            Maximum traversal depth allowed for generated paths.

        Returns
        -------
        List
            Internal traversal path representations.
        """

        paths = []

        for label in self.schema.labels:

            # depth 0 represents direct queries over a single node
            if depth == 0:
                paths.append([label])
                continue

            max_depth = get_max_depth(label, depth)

            def dfs(current_path, current_label, remaining_depth):

                # Any path containing at least one relationship traversal
                # is considered a valid relational structure
                if len(current_path) > 1:
                    paths.append(current_path)

                if remaining_depth == 0:
                    return

                allowed_targets = get_allowed_targets(current_label)

                for rel in self.schema.get_outgoing(current_label):

                    # Enforce schema-level restrictions on allowed traversal targets
                    if allowed_targets and rel.target not in allowed_targets:
                        continue

                    # Prevent repeated nodes when cyclic traversals are disabled
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

        Each path may produce different intent structures through combinations
        of supported query components:

        - return projections
        - filter constraints
        - aggregations
        - ordering
        - result limits

        Expansion follows regime policies to preserve structural diversity while
        limiting uncontrolled combinatorial growth.
        """

        intents = []

        target_candidates = self._get_target_candidates(path)

        for target_label in target_candidates:

            projections = self._expand_projections(target_label)

            # Ignore target nodes that cannot generate meaningful return structures.
            if not projections:
                continue

            max_proj = self.regime_policy[
                "max_projection_samples"
            ]

            if len(projections) > max_proj:
                projections = random.sample(
                    projections,
                    max_proj
                )

            for projection in projections:

                base_intent = IntentSpec(
                    intent=IntentCore(
                        regime=self.config.regime_name,
                        
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
        Generate valid return attribute combinations for a target node.

        Projection generation is delegated to attribute policies, allowing
        centralized control over available attributes, aggregation constraints,
        and projection sampling behavior.

        Parameters
        ----------
        label : str
            Target node label.

        aggregate : optional
            Aggregation context used to restrict compatible projections.
        """

        return get_returnable_attributes(label, aggregate)

    # ==================================================
    # FILTERS EXPANSION
    # ==================================================

    def _expand_filters_variants(self, intents):
        """
        Expand query intents with structurally valid filter configurations.

        Supported filter variations include:

        - single attribute constraints
        - multiple attribute constraints
        - operator compatibility validation
        - numeric interval consistency checks
        - mandatory filter enforcement

        Expansion is controlled by regime policies and configuration limits
        to balance structural coverage and combinatorial complexity.
        """

        all_variants = []

        for intent in intents:
            all_variants.extend(
                self._expand_filters_single(intent)
            )

        return all_variants


    def _expand_filters_single(self, intent: IntentSpec):
        """
        Generate structurally valid filter variants for a single intent.

        The expansion process preserves:

        - attribute/operator compatibility
        - uniqueness of attribute constraints
        - numeric range consistency
        - mandatory filter requirements

        Generated variants are controlled by regime-specific policies to
        maintain diversity while avoiding unnecessary structural explosion.
        """

        variants = []

        # Skip filter generation when disabled by regime policy.
        if not self._allow_optional_filters():
            return variants

        # Collect labels involved in the query structure.
        labels = self._get_all_labels(intent)

        # Generate candidate filters per node label.
        filter_candidates_by_label = (
            self._build_filter_candidates(labels)
        )

        # Relational queries require filters across path nodes.
        if self._is_relational_filter_generation(labels):

            return self._generate_relational_filter_variants(
                intent,
                labels,
                filter_candidates_by_label
            )

        # Generate single-filter variants.
        variants.extend(
            self._generate_single_filter_variants(
                intent,
                filter_candidates_by_label
            )
        )

        # Generate multi-filter variants when enabled.
        if self.config.allow_multiple_filters:

            variants.extend(
                self._generate_multi_filter_variants(
                    intent,
                    filter_candidates_by_label
                )
            )

        return variants


    def _allow_optional_filters(self):
        """
        Determine whether optional filter generation is enabled for
        the current query regime.

        Returns
        -------
        bool
            True when optional filters may be generated.
        """

        policy = FILTER_GENERATION_POLICY.get(
            self.config.regime_name,
            {}
        )

        return policy.get(
            "allow_optional_filters",
            True
        )


    def _build_filter_candidates(self, labels):
        """
        Build possible filter constraints for each node label.

        Candidate generation combines:

        - filterable schema attributes
        - compatible operators
        - sampled values

        The resulting candidates are later expanded into complete
        structural filter variants.
        """

        filter_candidates_by_label = {}

        for label in labels:

            attributes = get_filterable_attributes(label)

            filter_candidates = []

            for attr in attributes:

                operators = get_operators(
                    label,
                    attr
                )

                if not operators:
                    continue

                k = min(
                    len(operators),
                    self.config.max_operators_per_attribute
                )

                sampled_ops = random.sample(
                    operators,
                    k
                )

                for op in sampled_ops:

                    filter_candidates.append(
                        AttributeFilter(
                            node_label=label,
                            attribute=attr,
                            operator=op,
                            value=sample_filter_value(
                                attr,
                                label
                            ),
                        )
                    )

            filter_candidates_by_label[label] = filter_candidates

        return filter_candidates_by_label


    def _is_relational_filter_generation(self, labels):
        """
        Check whether the current regime requires relational filter expansion.

        Relational regimes apply filter generation across multiple nodes
        connected through traversal paths.
        """

        RELATIONAL_REGIMES = {
            "relational_lookup_query",
            "relational_count_query",
            "relational_aggregation_query",
            "relational_ranking_query",
        }

        return (
            self.config.regime_name in RELATIONAL_REGIMES
            and len(labels) > 1
        )


    def _generate_relational_filter_variants(
        self,
        intent,
        labels,
        filter_candidates_by_label
    ):
        """
        Generate filter variants for relational query structures.

        Each participating node in the traversal path must contribute valid
        filter constraints. Candidate combinations are sampled according to
        configured limits to control combinatorial growth.

        Returns
        -------
        List[IntentSpec]
            Intents containing valid relational filter configurations.
        """

        variants = []

        per_label_combos = []

        for label in labels:

            candidates = filter_candidates_by_label.get(
                label,
                []
            )

            if not candidates:
                return variants

            label_variants = []

            max_k = min(
                self.config.max_filters_per_node,
                len(candidates)
            )

            for k in range(1, max_k + 1):

                combos = list(
                    itertools.combinations(
                        candidates,
                        k
                    )
                )

                combos = self._limit_filter_combinations(
                    combos
                )

                label_variants.extend(combos)

            per_label_combos.append(
                label_variants
            )

        all_combos = list(
            itertools.product(
                *per_label_combos
            )
        )

        max_combos = self.regime_policy[
            "max_filter_combinations"
        ]

        if len(all_combos) > max_combos:

            all_combos = random.sample(
                all_combos,
                max_combos
            )

        for combo in all_combos:

            filters = []

            for label_filters in combo:
                filters.extend(label_filters)

            if not self._valid_filter_set(filters):
                continue

            new_intent = self._create_filtered_intent(
                intent,
                labels,
                filters
            )

            variants.append(new_intent)

        return variants


    def _generate_single_filter_variants(
        self,
        intent,
        filter_candidates_by_label
    ):
        """
        Generate query variants containing exactly one filter constraint.

        Each candidate filter produces an independent intent copy while
        preserving mandatory filter rules and structural modifiers.
        """

        variants = []

        for label, candidates in filter_candidates_by_label.items():

            for f in candidates:

                new_intent = intent.model_copy(
                    deep=True
                )

                filters = enforce_mandatory_filters(
                    label,
                    [f]
                )

                new_intent.schema_spec.filters.extend(
                    filters
                )

                self._add_modifier(
                    new_intent,
                    StructuralModifier.FILTER
                )

                variants.append(
                    new_intent
                )

        return variants


    def _generate_multi_filter_variants(
        self,
        intent,
        filter_candidates_by_label
    ):
        """
        Generate query variants containing multiple compatible filters.

        Filter combinations are validated before intent creation to ensure:

        - no duplicated attribute constraints
        - compatible numeric conditions
        - schema-valid filter structures
        """

        variants = []

        for label, candidates in filter_candidates_by_label.items():

            max_k = min(
                self.config.max_filters_per_node,
                len(candidates)
            )

            for k in range(2, max_k + 1):

                combos = list(
                    itertools.combinations(
                        candidates,
                        k
                    )
                )

                combos = self._limit_filter_combinations(
                    combos
                )

                for combo in combos:

                    if not self._valid_filter_set(combo):
                        continue

                    new_intent = intent.model_copy(
                        deep=True
                    )

                    filters = enforce_mandatory_filters(
                        label,
                        list(combo)
                    )

                    new_intent.schema_spec.filters.extend(
                        filters
                    )

                    self._add_modifier(
                        new_intent,
                        StructuralModifier.FILTER
                    )

                    variants.append(
                        new_intent
                    )

        return variants


    def _limit_filter_combinations(self, combos):
        """
        Limit generated filter combinations according to regime policy.

        Random sampling is applied when the number of possible combinations
        exceeds the configured maximum.
        """

        max_combos = self.regime_policy[
            "max_filter_combinations"
        ]

        if len(combos) > max_combos:

            return random.sample(
                combos,
                max_combos
            )

        return combos


    def _valid_filter_set(self, filters):
        """
        Validate a complete filter combination.

        Validation ensures that:

        - each attribute receives at most one constraint
        - numeric filters define compatible ranges
        """

        attrs = {
            (
                f.node_label,
                f.attribute
            )
            for f in filters
        }

        if len(attrs) != len(filters):
            return False

        return self._filters_are_compatible(filters)


    def _create_filtered_intent(
        self,
        intent,
        labels,
        filters
    ):
        """
        Create a new intent containing validated filter constraints.

        Mandatory filters are applied per node label before the structural
        filter modifier is registered.
        """

        new_intent = intent.model_copy(
            deep=True
        )

        final_filters = []

        for label in labels:

            node_filters = [
                f
                for f in filters
                if f.node_label == label
            ]

            final_filters.extend(
                enforce_mandatory_filters(
                    label,
                    node_filters
                )
            )

        new_intent.schema_spec.filters.extend(
            final_filters
        )

        self._add_modifier(
            new_intent,
            StructuralModifier.FILTER
        )

        return new_intent


    def _filters_are_compatible(self, filters):
        """
        Validate compatibility between numeric filter constraints.

        Numeric constraints are grouped by attribute and checked to ensure
        that generated intervals remain logically consistent.

        Example
        -------
        value >= 10 AND value <= 5 -> invalid
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
        Generate aggregation-based query variants according to
        the active structural regime.

        Supported aggregation behaviors include:

        - count operations
        - numerical aggregations over compatible attributes
        - aggregation preserving existing filters
        - aggregation without filter constraints

        Aggregation expansion follows regime rules to maintain semantic
        consistency between query intent and generated structure.
        """

        expanded = []

        count_regimes = {
            "simple_count_query",
            "relational_count_query",
        }

        aggregation_regimes = {
            "simple_aggregation_query",
            "relational_aggregation_query",
        }


        for intent in intents:

            regime_name = intent.intent.regime


            # Generate count aggregation variants.
            if regime_name in count_regimes:

                new_intent = intent.model_copy(
                    deep=True
                )

                new_intent.schema_spec.aggregate = AggregateSpec(
                    function="count",
                    attribute=None,
                )

                new_intent.schema_spec.return_attributes = []

                self._add_modifier(
                    new_intent,
                    StructuralModifier.COUNT
                )

                expanded.append(
                    new_intent
                )

                continue


            # Generate attribute-based aggregation variants.
            if regime_name in aggregation_regimes:

                target_label = intent.schema_spec.target["label"]

                attributes = get_aggregatable_attributes(
                    target_label
                )


                for attr in attributes:

                    if not is_aggregatable(attr):
                        continue


                    aggregate_functions = get_aggregate_functions(attr)


                    for fn in aggregate_functions:

                        if fn == "count":
                            continue


                        # Preserve existing filters in aggregation variant.
                        filtered_intent = intent.model_copy(
                            deep=True
                        )

                        filtered_intent.schema_spec.aggregate = AggregateSpec(
                            function=fn,
                            attribute=attr,
                        )

                        filtered_intent.schema_spec.return_attributes = []

                        self._add_modifier(
                            filtered_intent,
                            StructuralModifier.AGGREGATE
                        )

                        expanded.append(
                            filtered_intent
                        )


                        # Create aggregation variant without filters.
                        empty_filter_intent = intent.model_copy(
                            deep=True
                        )

                        empty_filter_intent.schema_spec.filters = []

                        empty_filter_intent.schema_spec.aggregate = AggregateSpec(
                            function=fn,
                            attribute=attr,
                        )

                        empty_filter_intent.schema_spec.return_attributes = []

                        self._add_modifier(
                            empty_filter_intent,
                            StructuralModifier.AGGREGATE
                        )

                        expanded.append(
                            empty_filter_intent
                        )


        return expanded


    # ==================================================
    # ORDER BY
    # ==================================================

    def _expand_order_by(self, intents):
        """
        Generate ordering variants for query intents.

        Ordering expansion considers:

        - attributes allowed for sorting
        - aggregation context compatibility
        - configured ordering directions
        - preservation or removal of existing filters

        Generated variants receive an ORDER_BY structural modifier.
        """

        expanded = []

        for intent in intents:

            label = intent.schema_spec.target["label"]

            attributes = get_orderable_attributes(label)

            # Prioritize aggregation attributes when ordering aggregated results.
            if intent.schema_spec.aggregate:

                agg_attr = intent.schema_spec.aggregate.attribute

                if agg_attr and agg_attr in attributes:

                    attributes = (
                        [agg_attr]
                        +
                        [
                            a
                            for a in attributes
                            if a != agg_attr
                        ]
                    )

            max_attrs = self.regime_policy[
                "max_order_attributes"
            ]

            # Limit explored ordering attributes according to regime policy.
            if max_attrs > 0 and len(attributes) > max_attrs:

                if intent.schema_spec.aggregate:

                    attributes = (
                        attributes[:1]
                        +
                        random.sample(
                            attributes[1:],
                            max_attrs - 1
                        )
                    )

                else:

                    attributes = random.sample(
                        attributes,
                        max_attrs
                    )


            for attr in attributes:

                for direction in self.config.order_by_directions:


                    # Create ORDER BY variant preserving filters.
                    filtered_intent = intent.model_copy(
                        deep=True
                    )

                    filtered_intent.schema_spec.order_by = {
                        "node_label": label,
                        "attribute": attr,
                        "direction": direction,
                    }

                    self._add_modifier(
                        filtered_intent,
                        StructuralModifier.ORDER_BY
                    )

                    expanded.append(
                        filtered_intent
                    )


                    # Create ORDER BY variant without filters.
                    if intent.schema_spec.filters:

                        empty_filter_intent = intent.model_copy(
                            deep=True
                        )

                        empty_filter_intent.schema_spec.filters = []

                        empty_filter_intent.schema_spec.order_by = {
                            "node_label": label,
                            "attribute": attr,
                            "direction": direction,
                        }

                        self._add_modifier(
                            empty_filter_intent,
                            StructuralModifier.ORDER_BY
                        )

                        expanded.append(
                            empty_filter_intent
                        )


        return expanded

    # ==================================================
    # LIMIT
    # ==================================================

    def _expand_limit(self, intents):
        """
        Generate query variants containing result size constraints.

        Limit values are sampled according to regime policies to control
        structural expansion while introducing different result cardinality
        patterns.
        """

        expanded = []

        for intent in intents:

            limit_values = self.config.limit_values

            max_limits = self.regime_policy[
                "max_limit_variants"
            ]

            if max_limits > 0 and len(limit_values) > max_limits:

                limit_values = random.sample(
                    limit_values,
                    max_limits
                )

            for value in limit_values:

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
        Register a structural modifier in an intent.

        Duplicate modifiers are ignored to preserve modifier consistency.
        """

        if modifier not in intent.intent.modifiers:
            intent.intent.modifiers.append(modifier)

    def _get_target_candidates(self, path):
        """
        Extract valid target node candidates from a traversal path.

        The target corresponds to the final node reached by the traversal
        structure.
        """

        last_step = path[-1]

        if isinstance(last_step, str):
            return [last_step]

        return [last_step[1]]

    def _build_path_spec(self, raw_path):
        """
        Convert internal traversal representations into the schema format
        used by IntentSpec.

        The generated representation describes relationships between source
        nodes and target nodes.
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
        Collect all node labels participating in a query intent.

        Labels are extracted from the target node and traversal path to
        identify all entities involved in structural expansion.
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

        Structural equality is determined from serialized intent
        representations, which act as fingerprints for duplicate detection.
        """

        seen = set()

        unique = []

        for intent in intents:

            key = intent.model_dump_json()

            if key not in seen:

                seen.add(key)

                unique.append(intent)

        return unique
