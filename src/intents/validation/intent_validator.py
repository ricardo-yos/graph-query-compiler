"""
Intent Validation Logic
=======================

Semantic validation layer for structured graph query intents.

This module validates whether generated intents are structurally
consistent, semantically meaningful and compatible with the
constraints defined by the intent generation pipeline.

Validation is performed declaratively using semantic definitions
provided by `intent_semantic_rules`.

Validation Scope
----------------
- intent metadata structure
- target node definitions
- return attribute visibility
- filter semantic compatibility
- ORDER BY constraints
- LIMIT constraints
- traversal path consistency
- aggregation validity
- regime/path compatibility

Validation Responsibilities
---------------------------
- prevent structurally invalid intents
- enforce semantic compatibility between attributes and values
- restrict unsupported aggregation behaviors
- ensure regime-specific structural consistency
- guarantee compatibility with natural-language query patterns

This module does NOT:
---------------------
- generate intents
- modify intent structures
- traverse graph schemas
- perform scoring or ranking
- sample filter values

Used By
-------
- structural dataset generation pipeline
- combinatorial intent generator
- semantic filtering stages
- dataset quality control

Dependencies
------------
intent_semantic_rules
    Declarative semantic definitions and attribute policies.
"""

from .intent_semantic_rules import (
    NL_VISIBLE_ATTRIBUTES,
    ALLOWED_COUNT_LABELS,
    ATTRIBUTE_VALUE_TYPES,
)


# --------------------------------------------------
# Global semantic constraints
# --------------------------------------------------

# Maximum number of traversal hops allowed in graph paths.
# Prevents overly complex query structures unlikely to appear
# naturally in NL graph querying scenarios.
MAX_PATH_LENGTH = 3

# Structural regimes that represent count queries.
COUNT_REGIMES = {
    "simple_count_query",
    "relational_count_query",
}

# Structural regimes that support numeric aggregations.
AGGREGATION_REGIMES = {
    "simple_aggregation_query",
    "relational_aggregation_query",
}

# Structural regimes representing direct retrieval queries.
LOOKUP_REGIMES = {
    "simple_lookup_query",
    "relational_lookup_query",
}

# Structural regimes that require ordering and limiting behavior.
RANKING_REGIMES = {
    "simple_ranking_query",
    "relational_ranking_query",
}


# --------------------------------------------------
# Internal helper
# --------------------------------------------------

def get_schema(intent: dict) -> dict:
    """
    Extract schema specification from structured intent object.

    Parameters
    ----------
    intent : dict
        Full structured intent.

    Returns
    -------
    dict
        Schema specification section.
    """
    return intent.get("schema_spec", {})


def get_intent_meta(intent: dict) -> dict:
    """
    Extract semantic intent metadata.

    Parameters
    ----------
    intent : dict
        Full structured intent.

    Returns
    -------
    dict
        Intent metadata section.
    """

    return intent.get("intent", {})


# --------------------------------------------------
# Attribute-level helpers
# --------------------------------------------------

def is_nl_visible(label: str, attribute: str) -> bool:
    """
    Check whether an attribute can appear explicitly in
    natural-language graph queries.

    Parameters
    ----------
    label : str
        Entity label.

    attribute : str
        Attribute name.

    Returns
    -------
    bool
        True if attribute is NL-visible.
    """
    return attribute in NL_VISIBLE_ATTRIBUTES.get(label, set())


def is_numeric_type(expected_type) -> bool:
    """
    Check whether semantic attribute type is numeric.

    Parameters
    ----------
    expected_type
        Semantic type definition.

    Returns
    -------
    bool
        True if numeric-compatible.
    """

    return expected_type in [(int, float), int, float]


def is_value_type_valid(label: str, attribute: str, value) -> bool:
    """
    Validate semantic compatibility between attribute and filter value.

    Supports lightweight coercion for common natural-language
    representations, including:

    - numeric values represented as strings
    - boolean values represented with NL terms

    Examples
    --------
    "4.5" -> float
    "yes" -> bool
    "true" -> bool

    Parameters
    ----------
    label : str
        Entity label.

    attribute : str
        Attribute name.

    value :
        Filter value.

    Returns
    -------
    bool
        True if value is semantically compatible.
    """

    expected_type = ATTRIBUTE_VALUE_TYPES.get(label, {}).get(attribute)

    if expected_type is None:
        return False

    if isinstance(value, expected_type):
        return True

    # numeric values expressed as strings
    if is_numeric_type(expected_type):

        if isinstance(value, str):

            try:
                float(value)
                return True

            except ValueError:
                return False

    # allow numeric values expressed as strings
    if expected_type in [(int, float), int, float]:
        if isinstance(value, str):
            try:
                float(value)
                return True
            except ValueError:
                return False

    # allow boolean values expressed in NL
    if expected_type is bool and isinstance(value, str):

        if value.lower() in {
            "true",
            "false",
            "1",
            "0",
            "yes",
            "no",
        }:
            return True

    return False


# --------------------------------------------------
# Intent metadata validation
# --------------------------------------------------

def validate_intent_meta(intent: dict) -> bool:
    """
    Validate high-level intent metadata structure.

    Checks:
    - metadata structure validity
    - regime type integrity
    - modifier list consistency

    Parameters
    ----------
    intent : dict
        Structured intent.

    Returns
    -------
    bool
        True if metadata is valid.
    """

    meta = get_intent_meta(intent)

    if not isinstance(meta, dict):
        return False

    regime = meta.get("regime")

    if regime is not None and not isinstance(regime, str):
        return False

    modifiers = meta.get("modifiers")

    if modifiers is not None:

        if not isinstance(modifiers, list):
            return False

        if not all(isinstance(m, str) for m in modifiers):
            return False

    return True


# --------------------------------------------------
# Target validation
# --------------------------------------------------

def validate_target(intent: dict) -> bool:
    """
    Validate target node structure.

    Checks:
    - target existence
    - target structural integrity
    - label validity

    Parameters
    ----------
    intent : dict
        Structured intent.

    Returns
    -------
    bool
        True if target structure is valid.
    """

    schema = get_schema(intent)

    target = schema.get("target")

    if not isinstance(target, dict):
        return False

    label = target.get("label")

    if not isinstance(label, str):
        return False

    return True


# --------------------------------------------------
# Return attributes validation
# --------------------------------------------------

def validate_return_attributes(intent: dict) -> bool:
    """
    Validate requested return attributes.

    Ensures:
    - return_attributes is structurally valid
    - attributes are NL-visible
    - attributes are compatible with target entity

    Parameters
    ----------
    intent : dict
        Structured intent.

    Returns
    -------
    bool
        True if return attributes are valid.
    """

    schema = get_schema(intent)

    label = schema.get("target", {}).get("label")
    attrs = schema.get("return_attributes", [])

    if not isinstance(attrs, list):
        return False

    if not all(isinstance(a, str) for a in attrs):
        return False

    for attr in attrs:
        if not is_nl_visible(label, attr):
            return False

    return True


# --------------------------------------------------
# Filter validation
# --------------------------------------------------

def validate_filters(intent: dict) -> bool:
    """
    Validate filter clause semantic consistency.

    Checks:
    - filter structural integrity
    - attribute existence
    - semantic type compatibility
    - filter value validity

    Parameters
    ----------
    intent : dict
        Structured intent.

    Returns
    -------
    bool
        True if filters are semantically valid.
    """

    schema = get_schema(intent)

    filters = schema.get("filters")

    if filters is None:
        return True

    if not isinstance(filters, list):
        return False

    for f in filters:

        label = f.get("node_label")
        attr = f.get("attribute")

        if not isinstance(label, str):
            return False

        if not isinstance(attr, str):
            return False

        if ATTRIBUTE_VALUE_TYPES.get(label, {}).get(attr) is None:
            return False

        value = (
            f.get("value")
            or f.get("value_int")
            or f.get("value_float")
            or f.get("value_str")
        )

        if value is not None:

            if not is_value_type_valid(label, attr, value):
                return False

    return True


# --------------------------------------------------
# Order by validation
# --------------------------------------------------

def validate_order_by(intent: dict) -> bool:
    """
    Validate ORDER BY clause consistency.

    Checks:
    - ranking regimes require ordering
    - ordering attribute existence
    - ordering direction validity
    - attribute compatibility

    Parameters
    ----------
    intent : dict
        Structured intent.

    Returns
    -------
    bool
        True if ORDER BY is valid.
    """

    schema = get_schema(intent)

    meta = get_intent_meta(intent)

    regime = meta.get("regime")

    order_by = schema.get("order_by")

    # ranking regimes require ordering
    if regime in RANKING_REGIMES:

        if order_by is None:
            return False

    if order_by is None:
        return True

    # current schema uses single dict
    if not isinstance(order_by, dict):
        return False

    label = schema.get("target", {}).get("label")

    attr = order_by.get("attribute")

    if not attr:
        return False

    if ATTRIBUTE_VALUE_TYPES.get(label, {}).get(attr) is None:
        return False

    direction = order_by.get("direction")

    if direction not in {"asc", "desc"}:
        return False

    return True


# --------------------------------------------------
# Limit validation
# --------------------------------------------------

def validate_limit(intent: dict) -> bool:
    """
    Validate LIMIT clause consistency.

    Checks:
    - ranking regimes require limits
    - limit type validity
    - positive integer constraints

    Parameters
    ----------
    intent : dict
        Structured intent.

    Returns
    -------
    bool
        True if LIMIT is valid.
    """

    schema = get_schema(intent)

    meta = get_intent_meta(intent)

    regime = meta.get("regime")

    limit = schema.get("limit")

    # ranking queries require limit
    if regime in RANKING_REGIMES:

        if limit is None:
            return False

    if limit is None:
        return True

    if not isinstance(limit, int):
        return False

    if limit <= 0:
        return False

    return True


# --------------------------------------------------
# Path validation
# --------------------------------------------------

def validate_path(intent: dict) -> bool:
    """
    Validate graph traversal path structure.

    Checks:
    - traversal depth constraints
    - path structural integrity
    - relationship field presence
    - target consistency

    Parameters
    ----------
    intent : dict
        Structured intent.

    Returns
    -------
    bool
        True if traversal path is valid.
    """

    schema = get_schema(intent)

    path = schema.get("path")

    if path is None:
        return True

    if not isinstance(path, list):
        return False

    if len(path) > MAX_PATH_LENGTH:
        return False

    for step in path:

        if "relationship" not in step:
            return False

        if "target" not in step:
            return False

    return True


# --------------------------------------------------
# Aggregate validation
# --------------------------------------------------

def validate_aggregate(intent: dict) -> bool:
    """
    Validate aggregate semantic compatibility.

    Supported aggregate behaviors:
    - COUNT(*)
    - SUM
    - AVG
    - MIN
    - MAX

    Checks:
    - regime compatibility
    - aggregate target validity
    - numeric aggregation constraints
    - COUNT semantics

    Parameters
    ----------
    intent : dict
        Structured intent.

    Returns
    -------
    bool
        True if aggregation is valid.
    """

    schema = get_schema(intent)

    meta = get_intent_meta(intent)

    regime = meta.get("regime")

    agg = schema.get("aggregate")

    # lookup regimes cannot aggregate
    if regime in LOOKUP_REGIMES.union(RANKING_REGIMES):

        if agg is not None:
            return False

        return True

    # aggregation regimes require aggregate
    if regime in COUNT_REGIMES.union(AGGREGATION_REGIMES):

        if agg is None:
            return False

    if agg is None:
        return True

    func = agg.get("function")
    attr = agg.get("attribute")

    label = schema.get("target", {}).get("label")

    if not func or not label:
        return False

    # --------------------------------------------------
    # COUNT
    # --------------------------------------------------

    if func == "count":

        # count only allowed in count regimes
        if regime not in COUNT_REGIMES:
            return False

        # COUNT(*)
        if attr is not None:
            return False

        if label not in ALLOWED_COUNT_LABELS:
            return False

        return True

    # --------------------------------------------------
    # STANDARD AGGREGATIONS
    # --------------------------------------------------

    if func in {"sum", "avg", "min", "max"}:

        # numeric aggregations only allowed in aggregation regimes
        if regime not in AGGREGATION_REGIMES:
            return False

        if attr is None:
            return False

        expected_type = ATTRIBUTE_VALUE_TYPES.get(
            label,
            {},
        ).get(attr)

        if not is_numeric_type(expected_type):
            return False

        return True

    return False


# --------------------------------------------------
# Regime/path consistency
# --------------------------------------------------

def validate_regime_path_consistency(intent: dict) -> bool:
    """
    Validate consistency between structural regime and traversal path.

    Rules
    -----
    - simple regimes cannot traverse graph
    - relational regimes must contain traversal paths

    Parameters
    ----------
    intent : dict
        Structured intent.

    Returns
    -------
    bool
        True if regime/path consistency is valid.
    """

    schema = get_schema(intent)

    meta = get_intent_meta(intent)

    regime = meta.get("regime")

    path = schema.get("path", [])

    has_path = len(path) > 0

    # simple regimes cannot traverse graph
    if regime and regime.startswith("simple"):

        if has_path:
            return False

    # relational regimes require traversal
    if regime and regime.startswith("relational"):

        if not has_path:
            return False

    return True


# --------------------------------------------------
# Main validation entry point
# --------------------------------------------------

def is_valid_intent(intent: dict) -> bool:
    """
    Run complete semantic validation pipeline for a structured intent.

    Sequentially validates:
    - intent metadata
    - target structure
    - return attributes
    - filters
    - ordering
    - limits
    - traversal paths
    - aggregations
    - regime consistency

    Parameters
    ----------
    intent : dict
        Structured intent.

    Returns
    -------
    bool
        True if intent passes all validation stages.
    """

    if not validate_intent_meta(intent):
        return False

    if not validate_target(intent):
        return False

    if not validate_return_attributes(intent):
        return False

    if not validate_filters(intent):
        return False

    if not validate_order_by(intent):
        return False

    if not validate_limit(intent):
        return False

    if not validate_path(intent):
        return False

    if not validate_aggregate(intent):
        return False

    if not validate_regime_path_consistency(intent):
        return False

    return True
