"""
Attribute Utility Layer
=======================

Helper functions that expose semantic policies used by the
structural intent generator.

This module abstracts policy dictionaries behind a stable API,
ensuring the generator does not directly depend on policy structure.

Responsibilities:
- attribute eligibility (filterable, orderable, returnable)
- operator validation
- aggregate validation
- filter value sampling
- enforcement of mandatory constraints

Design goals:
- decouple generator from policy files
- ensure consistent rule enforcement
- centralize semantic constraints
"""

import random
from typing import Optional
from ..intent_models import AttributeFilter

from ..policies.operator_policy import NODE_ATTRIBUTE_OPERATORS
from ..policies.aggregate_policy import AGGREGATE_FUNCTIONS
from ..policies.order_policy import ORDERABLE_ATTRIBUTES
from ..policies.return_policy import RETURN_POLICY
from ..policies.filter_policy import (
    FILTERABLE_ATTRIBUTES,
    FILTER_VALUE_RANGES,
    MANDATORY_FILTERS,
)
from ..policies.numeric_policy import NUMERIC_ATTRIBUTES
from ..policies.value_policy import VALUE_SAMPLES


# --------------------------------------------------
# Aggregate utilities
# --------------------------------------------------

def get_aggregate_functions(attribute: Optional[str]):
    """
    Return aggregate functions allowed for a given attribute.

    attribute=None corresponds to COUNT(*).
    """

    if attribute is None:
        return ["count"]

    return AGGREGATE_FUNCTIONS.get(attribute, [])


def is_aggregate_valid(attribute: Optional[str], function: str) -> bool:
    """
    Validate aggregate compatibility with attribute.
    """

    return function in get_aggregate_functions(attribute)


def is_aggregatable(attribute: Optional[str]) -> bool:
    """
    Check whether an attribute supports aggregation.
    """

    return len(get_aggregate_functions(attribute)) > 0


# --------------------------------------------------
# Operator utilities
# --------------------------------------------------

def get_operators(label: str, attribute: str):
    """
    Return allowed operators for a node attribute.

    Resolution order:
    1. node-specific policy
    2. general fallback policy
    """

    node_policy = NODE_ATTRIBUTE_OPERATORS.get(label, {})

    if attribute in node_policy:
        return node_policy[attribute]

    # fallback improves robustness when schema evolves
    general_policy = NODE_ATTRIBUTE_OPERATORS.get("General", {})

    return general_policy.get(attribute, [])


def is_operator_valid(label: str, attribute: str, operator: str) -> bool:
    """
    Validate operator compatibility with attribute.
    """

    return operator in get_operators(label, attribute)


# --------------------------------------------------
# Filter rules
# --------------------------------------------------

def get_filterable_attributes(label: str) -> list[str]:
    """
    Return attributes that support filtering for a node label.

    Some labels enforce mandatory attributes even if not declared
    explicitly in the base policy.
    """

    attrs = list(FILTERABLE_ATTRIBUTES.get(label, []))

    # mandatory semantic constraint
    if label == "Place" and "type" not in attrs:
        attrs.append("type")

    return attrs


def get_filter_values(attribute: str):
    """
    Return predefined candidate values for an attribute.
    """

    return FILTER_VALUE_RANGES.get(attribute)


def sample_filter_value(attribute: str, label: str):
    """
    Sample a valid filter value according to policy rules.

    Sampling ensures structural diversity while preserving
    semantic plausibility.
    """

    node_policy = FILTER_VALUE_RANGES.get(label, {})

    policy = node_policy.get(attribute)

    if not policy:
        policy = FILTER_VALUE_RANGES.get("General", {}).get(attribute)

    if not policy:
        return _safe_default_value(attribute)

    if policy["type"] == "range":

        # numeric attributes use bounded random sampling
        return random.randint(policy["min"], policy["max"])

    if policy["type"] in ("discrete", "categorical"):

        return random.choice(policy["values"])

    return _safe_default_value(attribute)


def enforce_mandatory_filters(label: str, filters: list):
    """
    Ensure required filters are present for specific node types.

    Example:
    Place nodes may require a 'type' constraint when filtered.
    """

    if not filters:
        return filters

    required_rules = MANDATORY_FILTERS.get(label)

    if not required_rules:
        return filters

    existing_attrs = {f.attribute for f in filters}

    new_filters = list(filters)

    for rule in required_rules:

        attr = rule["attribute"]

        if attr in existing_attrs:
            continue

        new_filters.append(

            AttributeFilter(
                node_label=label,
                attribute=attr,
                operator=rule["operator"],
                value=sample_filter_value(attr, label),
            )

        )

    return new_filters


def _safe_default_value(attribute: str, label: str):
    """
    Fallback value generator when no policy is defined.

    Heuristics rely on attribute naming conventions
    to avoid invalid structural combinations.
    """

    attr = attribute.lower()

    # numeric heuristics

    if "rate" in attr:
        return 70

    if "income" in attr:
        return 3000

    if "population" in attr:
        return 10000

    if "count" in attr:
        return 50

    if "length" in attr:
        return 500

    if "speed" in attr:
        return 50

    # text heuristics

    if "name" in attr:
        return "Unknown"

    if "type" in attr:
        return "unknown"

    # simple year fallback

    if "date" in attr:
        return 2022

    # ultimate fallback avoids None values in schema

    return 0


def is_numeric(attribute: str) -> bool:
    """
    Check whether attribute is numeric.
    """

    return attribute in NUMERIC_ATTRIBUTES


# --------------------------------------------------
# Ordering rules
# --------------------------------------------------

def is_orderable(label: str, attribute: str) -> bool:
    """
    Check if attribute supports ORDER BY for a label.
    """

    return attribute in ORDERABLE_ATTRIBUTES.get(label, set())


def get_orderable_attributes(label: str):
    """
    Return attributes eligible for ordering.
    """

    return list(ORDERABLE_ATTRIBUTES.get(label, []))


# --------------------------------------------------
# Projection rules
# --------------------------------------------------

def get_returnable_attributes(label: str, aggregate=None):
    """
    Return valid attribute projections for a node label.

    Output format:
        list[list[str]]

    Each inner list represents a valid projection combination.
    """

    policy = RETURN_POLICY.get(label)

    # fallback prevents empty projections
    if not policy:
        return [["name"]]

    # aggregate queries return aggregated value only
    if aggregate is not None:

        func = aggregate["function"]

        # COUNT(*) has no attribute projection
        if func == "count":
            return [[]]

        return [[aggregate["attribute"]]]

    projections = [policy["primary"]]

    # allow additional attributes when policy permits
    if policy["allow_multi"]:
        projections.extend(policy["secondary"])

    return projections
