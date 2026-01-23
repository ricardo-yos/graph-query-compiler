"""
Semantic Operator Policy
========================

Core policy layer responsible for determining which semantic
operators are valid for a given attribute.

This module defines a deterministic mapping between attribute
semantic classes (e.g., numeric, categorical, geographic) and
their allowed comparison or semantic operators.

It is used as a guardrail across multiple pipeline stages,
including:

- Intent derivation
- Intent validation
- Query construction
- Filtering logic

By centralizing operator constraints, this module ensures that
only semantically valid operations are produced (e.g., preventing
numeric comparisons on text attributes).

Important Characteristics
-------------------------
- Pure Python (no external dependencies)
- Deterministic and rule-based
- Schema-agnostic
- Safe defaults for unknown attributes

Notes
-----
- Unknown attributes default to the "categorical" semantic class.
- This policy is intentionally conservative to avoid invalid
  or nonsensical query generation.
"""

from typing import List


# --------------------------------------------------
# Semantic operator policy
# --------------------------------------------------

OPERATOR_POLICY = {
    "identifier": ["="],
    "name": ["="],
    "categorical": ["="],
    "numeric": ["=", ">", "<"],
    "geo": ["near"],
    "text": ["contains"],
}


# --------------------------------------------------
# Attribute classification rules
# --------------------------------------------------

ATTRIBUTE_CLASSES = {

    # -----------------------------
    # Identifiers (technical keys)
    # -----------------------------
    "identifier": {
        "place_id",
        "review_id",
        "osmid",
        "road_id",
        "neighborhood_id",
        "u",
        "v",
    },

    # -----------------------------
    # Proper names
    # -----------------------------
    "name": {
        "name",
        "author",
    },

    # -----------------------------
    # Numeric attributes
    # -----------------------------
    "numeric": {
        # Place
        "rating",
        "num_reviews",

        # Road
        "length",
        "maxspeed",

        # Neighborhood
        "area_km2",
        "average_monthly_income",
        "literacy_rate",
        "population_with_income",
        "total_literate_population",
        "total_private_households",
        "total_resident_population",

        # Intersection
        "street_count",
    },

    # -----------------------------
    # Categorical / enum-like
    # -----------------------------
    "categorical": {
        # Place
        "type",

        # Road / Intersection
        "highway",
        "oneway",
    },

    # -----------------------------
    # Geographic attributes
    # -----------------------------
    "geo": {
        # Place
        "latitude",
        "longitude",

        # Neighborhood
        "centroid_lat",
        "centroid_lon",

        # Intersection
        "lat",
        "lon",
    },

    # -----------------------------
    # Free text
    # -----------------------------
    "text": {
        "text",
    },
}


# --------------------------------------------------
# Attribute resolution utilities
# --------------------------------------------------

def get_attribute_class(attribute: str) -> str:
    """
    Resolve the semantic class associated with a given attribute.

    Parameters
    ----------
    attribute : str
        Attribute name to be classified.

    Returns
    -------
    str
        Semantic class name.
        Defaults to "categorical" if the attribute is unknown.

    Notes
    -----
    - Classification is deterministic and rule-based.
    - Unknown attributes fall back to a safe default to prevent
      invalid operator usage.
    """
    for semantic_class, attributes in ATTRIBUTE_CLASSES.items():
        if attribute in attributes:
            return semantic_class

    return "categorical"


def get_operators(attribute: str) -> List[str]:
    """
    Retrieve the list of valid operators for a given attribute.

    The allowed operators are determined by the semantic class
    resolved for the attribute.

    Parameters
    ----------
    attribute : str
        Attribute name for which operators should be retrieved.

    Returns
    -------
    List[str]
        List of allowed operators.

    Notes
    -----
    - Acts as a central constraint mechanism for intent generation
      and query validation.
    - Guarantees semantic correctness at the operator level.
    """
    semantic_class = get_attribute_class(attribute)
    return OPERATOR_POLICY[semantic_class]
