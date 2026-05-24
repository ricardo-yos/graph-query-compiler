"""
Intent Semantic Rules
=====================

Declarative semantic configuration for structural graph query intents.

This module centralizes domain-level semantic constraints and
natural-language visibility rules used across the intent pipeline.

The definitions provided here support:

- semantic validation
- operator compatibility policies
- natural-language query generation
- attribute visibility constraints
- semantic consistency enforcement

Semantic Configuration Includes
-------------------------------
- natural-language visible attributes
- semantically valid count targets
- attribute semantic value types

Design Principles
-----------------
- declarative-only configuration
- centralized semantic definitions
- schema-independent semantic policies
- reusable validation metadata
- pipeline-wide semantic consistency

Notes
-----
This module intentionally contains no executable validation logic,
schema traversal behavior or structural query generation.

All semantic enforcement is implemented by validator and policy
components elsewhere in the pipeline.

Dependencies
------------
None (pure declarative configuration module)
"""


# --------------------------------------------------
# Natural-language visible attributes
# --------------------------------------------------

# Attributes that can naturally appear in explicit
# natural-language graph queries.
NL_VISIBLE_ATTRIBUTES = {

    "Place": {
        "name",
        "type",
        "rating",
        "num_reviews",
    },

    "Neighborhood": {
        "name",
        "area_km2",
        "average_monthly_income",
        "total_resident_population",
    },

    "Road": {
        "name",
        "highway",
        "maxspeed",
        "oneway",
    },

    "Review": {
        "rating",
        "text",
        "author",
        "date",
    },

    # Intersections are rarely referenced directly in NL,
    # but connectivity-related properties may still appear
    # in structural graph queries.
    "Intersection": {
        "street_count",
    },
}


# --------------------------------------------------
# Allowed count targets
# --------------------------------------------------

# Entity labels for which count-style questions are
# semantically meaningful in natural language.
ALLOWED_COUNT_LABELS = {
    "Place",
    "Neighborhood",
    "Review",
    "Road",
    "Intersection",
}


# --------------------------------------------------
# Attribute value semantic types
# --------------------------------------------------

# Defines expected semantic value types for attributes.
#
# Used by semantic validators and operator policies to
# enforce compatibility between:
# - attributes
# - operators
# - sampled filter values
ATTRIBUTE_VALUE_TYPES = {

    "Place": {
        "name": str,
        "type": str,
        "rating": (int, float),
        "num_reviews": int,
    },

    "Neighborhood": {
        "name": str,
        "area_km2": (int, float),
        "average_monthly_income": (int, float),
        "total_resident_population": int,
    },

    "Road": {
        "name": str,
        "highway": str,
        "maxspeed": (int, float),
        "oneway": bool,
    },

    "Review": {
        "rating": (int, float),
        "text": str,
        "author": str,
        # stored as string to support flexible natural-language
        # temporal expressions and partial date references
        "date": str,
    },

    "Intersection": {
        "street_count": int,
    },
}
