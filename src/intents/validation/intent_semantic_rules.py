"""
Intent Semantic Rules
=====================

Declarative rule set defining which attributes, entities, and
intent components are considered valid and meaningful for
Natural Language (NL) interaction.

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

All structural and semantic enforcement is handled by
validator and policy components in the pipeline.

Dependencies
------------
- None (pure configuration module)
"""


# --------------------------------------------------
# Natural-language visible attributes
# --------------------------------------------------

# Attributes that can reasonably appear explicitly
# in Natural Language (NL) queries.
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
        "rating",
    },

    "Neighborhood": {
        "name",
        "total_resident_population",
    },

    "Road": {
        "name",
        "highway",
        "maxspeed",
        "oneway",
    },

    # Reviews are generally unknown beforehand,
    # except possibly by author name
    "Review": {
        "author",
        "rating",
    },

    "Intersection": {
        "street_count",
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
    "Intersection",
}


# --------------------------------------------------
# Attribute value semantic types
# --------------------------------------------------

# Defines the expected semantic type of attribute values.
# Used by validators and operator policies to ensure
# semantic consistency between attributes and operators.
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
        # stored as string to allow flexible NL temporal expressions
        "date": str,
    },

    "Intersection": {
        "street_count": int,
    },
}
