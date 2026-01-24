"""
Intent Semantic Rules
=====================

Declarative rule set defining which attributes, entities, and
intent components are considered valid and meaningful for
natural language interaction.

This module centralizes domain-level semantic knowledge used
across the intent pipeline, including:
- Natural-language visibility constraints
- Human-known attribute definitions
- Allowed count targets
- Attribute value semantic types

This file is intentionally declarative:
- No executable logic
- No validation code
- No schema traversal

All structural and semantic enforcement is handled by the
intent validator and related pipeline components.

Dependencies
------------
- None (pure configuration module)
"""


# --------------------------------------------------
# Natural-language visible attributes
# --------------------------------------------------

# Attributes that can reasonably appear explicitly
# in natural language queries.
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

    # Intersections are rarely mentioned directly,
    # but connectivity-related properties may be queried
    "Intersection": {
        "street_count",
    },
}


# --------------------------------------------------
# Human-known attributes
# --------------------------------------------------

# Attributes that a user is realistically expected
# to know or provide at query time.
HUMAN_KNOWN_ATTRIBUTES = {

    "Place": {
        "name",
        "type",
    },

    "Neighborhood": {
        "name",
    },

    "Road": {
        "name",
        "highway",
    },

    # Reviews are generally unknown beforehand,
    # except possibly by author name
    "Review": {
        "author",
    },
}


# --------------------------------------------------
# Allowed count targets
# --------------------------------------------------

# Entity labels for which count-style questions
# are meaningful and natural in NL.
ALLOWED_COUNT_LABELS = {
    "Place",
    "Neighborhood",
    "Review",
    "Road",
}


# --------------------------------------------------
# Attribute value semantic types
# --------------------------------------------------

# Defines the expected semantic type of attribute values.
# Used by validators and operator policies.
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
        "date": str,  # kept as str to allow flexible NL parsing
    },

    "Intersection": {
        "street_count": int,
    },
}
