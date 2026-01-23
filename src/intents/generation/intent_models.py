"""
Intent Specification Models
===========================

Canonical data models defining the structure of query intents
used throughout the intent derivation, validation, and dataset
generation pipelines.

These models act as the formal contract between all intent-related
stages, ensuring consistency, type safety, and extensibility.

Dependencies
------------
- typing
- pydantic

Notes
-----
- The models intentionally support both legacy and modern
  representations to allow gradual pipeline evolution.
- Flexibility is preserved where needed (e.g., filters) to avoid
  premature over-constraining during experimentation.
"""

from typing import List, Optional, Dict, Any

from pydantic import BaseModel


# --------------------------------------------------
# Filter models
# --------------------------------------------------

class AttributeFilter(BaseModel):
    """
    Atomic attribute-level filter definition.

    This model represents a single filtering condition applied
    to a node attribute within a graph-based query intent.

    Attributes
    ----------
    node_label : str
        Graph node label to which the filter applies.
    attribute : str
        Attribute name on the node.
    operator : str
        Operator defining the comparison or semantic constraint
        (e.g., '=', '>', '<', 'contains').
    value : Any
        Value used in the filtering condition.
    """

    node_label: str
    attribute: str
    operator: str
    value: Any


# --------------------------------------------------
# Core intent specification model
# --------------------------------------------------

class IntentSpec(BaseModel):
    """
    Canonical representation of a structured query intent.

    This model supports two structural paradigms:

    1. Legacy representation:
       - Simple, label-based intent definition
       - Preserved for backward compatibility

    2. Expressive representation:
       - Graph-aware intent composition
       - Explicit traversal paths, constraints, and targets

    Attributes
    ----------
    intent_type : str
        High-level classification of the intent
        (e.g., list, count, filter, rank).

    Legacy fields
    -------------
    start_label : Optional[str]
        Starting node label for legacy intent definitions.
    relationship_path : Optional[List[str]]
        Ordered list of relationship types (legacy format).
    end_label : Optional[str]
        Terminal node label for legacy intent definitions.

    Expressive fields
    -----------------
    known_inputs : Optional[List[Dict[str, Any]]]
        Predefined entities or constraints already known
        at query time.
    target : Optional[Dict[str, Any]]
        Target node or attributes the intent aims to retrieve.
    path : Optional[List[Dict[str, Any]]]
        Explicit graph traversal path with structured semantics.
    filters : List[Any]
        Collection of filtering constraints applied to the intent.

    Advanced fields
    ---------------
    order_by : Optional[Dict[str, Any]]
        Sorting definition for the query results.
    limit : Optional[int]
        Maximum number of results to return.

    Notes
    -----
    - The `filters` field is intentionally flexible to support
      multiple filter schemas during pipeline evolution.
    - This model acts as the central contract between all
      intent-related pipeline stages.
    """

    intent_type: str

    # ----------------------------------
    # Legacy model (backward compatibility)
    # ----------------------------------
    start_label: Optional[str] = None
    relationship_path: Optional[List[str]] = None
    end_label: Optional[str] = None

    # ----------------------------------
    # Expressive graph-aware model
    # ----------------------------------
    known_inputs: Optional[List[Dict[str, Any]]] = None
    target: Optional[Dict[str, Any]] = None
    path: Optional[List[Dict[str, Any]]] = None
    filters: List[Any] = []

    # ----------------------------------
    # Advanced query controls
    # ----------------------------------
    order_by: Optional[Dict[str, Any]] = None
    limit: Optional[int] = None
