"""
Intent Validation Logic
=======================

Semantic validation layer responsible for filtering structured
intents before they are accepted into downstream pipeline stages.

This module ensures that generated intents are:

- structurally consistent
- semantically meaningful
- compatible with Natural Language (NL) expression
- aligned with domain constraints

Validation is performed declaratively using rules defined in
`intent_semantic_rules`.

Scope of validation
-------------------
- target structure
- return attributes
- filter definitions
- ordering clauses
- aggregation rules
- path constraints
- attribute semantic types

This module does NOT:
- generate intents
- modify intents
- traverse graph schema
- perform scoring or ranking

Used by
-------
- dataset generation pipeline
- structural generator filtering stage
- intent quality control

Dependencies
------------
intent_semantic_rules :
    Declarative semantic definitions used for validation.
"""

from .intent_semantic_rules import (
    NL_VISIBLE_ATTRIBUTES,
    HUMAN_KNOWN_ATTRIBUTES,
    ALLOWED_COUNT_LABELS,
    ATTRIBUTE_VALUE_TYPES,
)


# --------------------------------------------------
# Global semantic constraints
# --------------------------------------------------

# Maximum number of relationship hops allowed in path traversal.
# Prevents overly complex graph queries unlikely to appear in NL.
MAX_PATH_LENGTH = 3


# --------------------------------------------------
# Internal helper
# --------------------------------------------------

def get_schema(intent: dict) -> dict:
    """
    Extract schema specification from intent.

    Parameters
    ----------
    intent : dict
        Full intent object.

    Returns
    -------
    dict
        Schema specification section of the intent.
    """
    return intent.get("schema_spec", {})


# --------------------------------------------------
# Attribute-level helpers
# --------------------------------------------------

def is_nl_visible(label: str, attribute: str) -> bool:
    """
    Check whether attribute can appear explicitly in NL queries.

    Parameters
    ----------
    label : str
        Node label.
    attribute : str
        Attribute name.

    Returns
    -------
    bool
    """
    return attribute in NL_VISIBLE_ATTRIBUTES.get(label, set())


def is_human_known(label: str, attribute: str) -> bool:
    """
    Check whether attribute is realistically known by the user.

    Parameters
    ----------
    label : str
        Node label.
    attribute : str
        Attribute name.

    Returns
    -------
    bool
    """
    return attribute in HUMAN_KNOWN_ATTRIBUTES.get(label, set())


def is_value_type_valid(label: str, attribute: str, value) -> bool:
    """
    Validate semantic compatibility between attribute and value.

    Allows light coercion for common NL representations:
    - numeric values as strings ("4.5")
    - boolean values as strings ("true", "yes")

    Parameters
    ----------
    label : str
        Node label.
    attribute : str
        Attribute name.
    value :
        Value provided in filter.

    Returns
    -------
    bool
    """

    expected_type = ATTRIBUTE_VALUE_TYPES.get(label, {}).get(attribute)

    if expected_type is None:
        return False

    if isinstance(value, expected_type):
        return True

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
        if value.lower() in {"true", "false", "1", "0", "yes", "no"}:
            return True

    return False


# --------------------------------------------------
# Intent metadata validation
# --------------------------------------------------

def validate_intent_meta(intent: dict) -> bool:
    """
    Validate high-level intent metadata structure.

    Ensures presence and type correctness of:
    - intent type
    - modifiers

    Parameters
    ----------
    intent : dict

    Returns
    -------
    bool
    """

    meta = intent.get("intent")

    if not isinstance(meta, dict):
        return False

    intent_type = meta.get("type")

    if not isinstance(intent_type, str):
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
    Validate presence and structure of target node.

    Parameters
    ----------
    intent : dict

    Returns
    -------
    bool
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
    Validate attributes requested in SELECT clause.

    Ensures attributes are NL-visible.

    Parameters
    ----------
    intent : dict

    Returns
    -------
    bool
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
    - attribute existence
    - semantic value compatibility
    - structural integrity

    Parameters
    ----------
    intent : dict

    Returns
    -------
    bool
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
        value = f.get("value")

        if not label or not attr:
            return False

        if ATTRIBUTE_VALUE_TYPES.get(label, {}).get(attr) is None:
            return False

        if value is not None:
            if not is_value_type_valid(label, attr, value):
                return False

        else:

            if not isinstance(label, str):
                return False

            if not isinstance(attr, str):
                return False

    return True


# --------------------------------------------------
# Order by validation
# --------------------------------------------------

def validate_order_by(intent: dict) -> bool:
    """
    Validate ORDER BY clause attributes.

    Parameters
    ----------
    intent : dict

    Returns
    -------
    bool
    """

    schema = get_schema(intent)

    order_by = schema.get("order_by")

    if order_by is None:
        return True

    if not isinstance(order_by, list):
        return False

    for ob in order_by:

        label = ob.get("node_label")
        attr = ob.get("attribute")

        if not label or not attr:
            return False

        if ATTRIBUTE_VALUE_TYPES.get(label, {}).get(attr) is None:
            return False

    return True


# --------------------------------------------------
# Limit validation
# --------------------------------------------------

def validate_limit(intent: dict) -> bool:
    """
    Validate LIMIT clause constraints.

    Parameters
    ----------
    intent : dict

    Returns
    -------
    bool
    """

    schema = get_schema(intent)

    limit = schema.get("limit")

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

    Constraints:
    - maximum path length
    - required keys per step

    Parameters
    ----------
    intent : dict

    Returns
    -------
    bool
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
    Validate aggregate function semantic compatibility.

    Supported functions:
    - count
    - sum
    - avg

    Parameters
    ----------
    intent : dict

    Returns
    -------
    bool
    """

    schema = get_schema(intent)
    agg = schema.get("aggregate")

    if agg is None:
        return True

    func = agg.get("function")
    attr = agg.get("attribute")
    label = schema.get("target", {}).get("label")

    if not func or not label:
        return False

    # COUNT
    if func == "count":
        if label not in ALLOWED_COUNT_LABELS:
            return False
        return True

    # SUM / AVG
    elif func in {"sum", "avg"}:
        if attr is None:
            return False

        expected_type = ATTRIBUTE_VALUE_TYPES.get(label, {}).get(attr)

        if expected_type not in [(int, float), int, float]:
            return False

        return True

    return False


# --------------------------------------------------
# Main validation entry point
# --------------------------------------------------

def is_valid_intent(intent: dict) -> bool:
    """
    Perform full semantic validation of an intent.

    Sequentially applies all validation checks.

    Parameters
    ----------
    intent : dict

    Returns
    -------
    bool
        True if intent is valid.
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

    return True
