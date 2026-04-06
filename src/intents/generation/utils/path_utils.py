"""
Path Utility Layer
==================

Helper functions that expose graph traversal constraints
defined in PATH_POLICY.

This module abstracts path-related policies from the structural
generator, ensuring that traversal rules remain centralized and
easy to evolve.

Responsibilities:
- restrict reachable node types
- control traversal depth per label
- control cycle allowance

Design goals:
- prevent invalid graph paths
- control combinatorial explosion
- keep generator independent from raw policy dictionaries
"""

from ..policies.path_policy import PATH_POLICY


def get_allowed_targets(label: str):
    """
    Return allowed target labels for outgoing relationships.

    If no policy is defined, returns an empty set,
    meaning all schema-defined relationships are allowed.
    """

    return PATH_POLICY.get(label, {}).get("allowed_targets", set())


def get_max_depth(label: str, default: int):
    """
    Return maximum traversal depth for a given label.

    Label-specific depth limits override global defaults,
    enabling fine-grained control over path expansion.
    """

    return PATH_POLICY.get(label, {}).get("max_depth", default)


def allow_cycles(label: str):
    """
    Check whether cyclic paths are allowed for a label.

    Cycles can significantly increase structural combinations,
    so this is typically disabled except for specific cases.
    """

    return PATH_POLICY.get(label, {}).get("allow_cycles", False)
