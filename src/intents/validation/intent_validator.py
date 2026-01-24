"""
Intent Validation Logic
======================

Core semantic validation rules for filtering intents before they are
accepted into downstream pipelines such as scoring, instruction
generation, and query synthesis.

This validator enforces:
- Natural-language visible attributes only
- Human-known attributes where applicable
- Semantic type consistency for filter values
- Structural consistency between path, return, and constraints

The goal is to guarantee that every intent:
- Can be naturally expressed by a human user
- Is semantically coherent with the graph schema
- Is safe for dataset generation and LLM fine-tuning
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

# Maximum traversal depth to keep queries realistic in NL
MAX_PATH_LENGTH = 3


# --------------------------------------------------
# Attribute-level helpers
# --------------------------------------------------

def is_nl_visible(label: str, attribute: str) -> bool:
    """
    Check whether an attribute is visible and meaningful in
    natural language queries.

    Parameters
    ----------
    label : str
        Node label (e.g., Place, Neighborhood).
    attribute : str
        Attribute name.

    Returns
    -------
    bool
        True if attribute is allowed to appear in NL questions.
    """
    return attribute in NL_VISIBLE_ATTRIBUTES.get(label, set())


def is_human_known(label: str, attribute: str) -> bool:
    """
    Check whether an attribute is typically known or explicitly
    provided by a human user.

    Examples
    --------
    - Place.name
    - Neighborhood.name

    Parameters
    ----------
    label : str
        Node label.
    attribute : str
        Attribute name.

    Returns
    -------
    bool
        True if attribute is considered human-known.
    """
    return attribute in HUMAN_KNOWN_ATTRIBUTES.get(label, set())


def is_value_type_valid(label: str, attribute: str, value) -> bool:
    """
    Validate whether a filter value matches the expected
    semantic type of the given label + attribute.

    This prevents semantically invalid filters such as:
    - rating = "high"
    - population > "many"

    Parameters
    ----------
    label : str
        Node label.
    attribute : str
        Attribute name.
    value : Any
        Value used in the filter.

    Returns
    -------
    bool
        True if value matches the expected semantic type.
    """
    expected_type = ATTRIBUTE_VALUE_TYPES.get(label, {}).get(attribute)

    if expected_type is None:
        return False

    return isinstance(value, expected_type)


# --------------------------------------------------
# Section-level validation
# --------------------------------------------------

def validate_return_section(intent: dict) -> bool:
    """
    Validate the 'return' section of the intent.

    Rules
    -----
    - Return section must exist
    - Return label must be defined
    - All returned attributes must be NL-visible
    """
    ret = intent.get("return")
    if not ret:
        return False

    label = ret.get("label")
    if not label:
        return False

    for attr in ret.get("attributes", []):
        if not is_nl_visible(label, attr):
            return False

    return True


def validate_known_section(intent: dict) -> bool:
    """
    Validate the 'known' section of the intent.

    Rules
    -----
    - Known attributes must be human-known
    - If 'known' is missing or empty, validation passes
    """
    known = intent.get("known")
    if not known:
        return True

    for item in known:
        label = item.get("label")
        attr = item.get("attribute")

        if not label or not attr:
            return False

        if not is_human_known(label, attr):
            return False

    return True


def validate_constraints(intent: dict) -> bool:
    """
    Validate the 'constraints' section, including:
    - filters
    - order_by
    - limit

    This step enforces semantic correctness and type safety.
    """
    constraints = intent.get("constraints", {})

    # -------------------------------
    # Filters
    # -------------------------------
    for f in constraints.get("filters", []):
        label = f.get("label")
        attr = f.get("attribute")
        value = f.get("value")

        if not label or not attr:
            return False

        if not is_nl_visible(label, attr):
            return False

        if value is not None:
            if not is_value_type_valid(label, attr, value):
                return False

    # -------------------------------
    # Order by
    # -------------------------------
    for ob in constraints.get("order_by", []):
        label = ob.get("label")
        attr = ob.get("attribute")

        if not label or not attr:
            return False

        if not is_nl_visible(label, attr):
            return False

    # -------------------------------
    # Limit
    # -------------------------------
    limit = constraints.get("limit")

    if limit is not None:
        if isinstance(limit, list):
            if not all(isinstance(l, int) and l > 0 for l in limit):
                return False
        elif not isinstance(limit, int) or limit <= 0:
            return False

    return True


def validate_path_and_constraints(intent: dict) -> bool:
    """
    Validate the interaction between path and constraints.

    Rules
    -----
    - Path length must be <= MAX_PATH_LENGTH
    - order_by labels must appear in path or return section
    """
    path = intent.get("path", [])
    constraints = intent.get("constraints", {})

    if len(path) > MAX_PATH_LENGTH:
        return False

    valid_nodes = set()
    for step in path:
        if step.get("from"):
            valid_nodes.add(step["from"])
        if step.get("to"):
            valid_nodes.add(step["to"])

    return_label = intent.get("return", {}).get("label")
    if return_label:
        valid_nodes.add(return_label)

    for ob in constraints.get("order_by", []):
        if ob.get("label") not in valid_nodes:
            return False

    return True


# --------------------------------------------------
# Aggregate validation entry point
# --------------------------------------------------

def is_valid_intent(intent: dict) -> bool:
    """
    Validate an intent according to the final dataset contract.

    This function aggregates all validation layers and acts as
    the single entry point for intent filtering.

    Returns
    -------
    bool
        True if intent is fully valid and safe for downstream usage.
    """
    return (
        validate_return_section(intent)
        and validate_known_section(intent)
        and validate_constraints(intent)
        and validate_path_and_constraints(intent)
    )
