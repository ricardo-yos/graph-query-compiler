"""
Intent Specification Models
===========================

Canonical data models representing structured graph query intents.

These models define the structural contract shared across:

- combinatorial intent generation
- natural language generation pipelines
- semantic parsing systems
- dataset serialization and validation

Design Goals
------------
- enforce schema consistency
- prevent silent schema drift
- support deterministic dataset generation
- preserve backward compatibility with existing datasets
- centralize structural query representation

Notes
-----
These models represent structural query intent specifications,
not executable queries or natural language text.
"""

from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict


# --------------------------------------------------
# Enums
# --------------------------------------------------

class StructuralModifier(str, Enum):
    """
    Structural operators applied to a base retrieval intent.

    Modifiers describe how the query structure is expanded beyond
    simple entity retrieval.
    """

    FILTER = "filter"
    AGGREGATE = "aggregate"
    COUNT = "count"
    ORDER_BY = "order_by"
    LIMIT = "limit"


# --------------------------------------------------
# Filter model
# --------------------------------------------------

class AttributeFilter(BaseModel):
    """
    Represents an attribute constraint applied to a graph node.

    Examples
    --------
    Movie.rating > 8
    Person.birth_year >= 1990
    Business.city = "Santo André"

    Notes
    -----
    The value field supports flexible typing to accommodate
    numeric, categorical and textual constraints.
    """

    model_config = ConfigDict(extra="forbid")

    node_label: str

    attribute: str

    operator: str

    # flexible typing supports numeric, string, categorical values
    value: Any


# --------------------------------------------------
# Aggregate model
# --------------------------------------------------

class AggregateSpec(BaseModel):
    """
    Represents an aggregation operation applied to query results.

    Examples
    --------
    avg(Movie.rating)
    count(*)
    max(Movie.revenue)

    Notes
    -----
    A null attribute supports global aggregations such as COUNT(*).
    """

    model_config = ConfigDict(extra="forbid")

    function: str

    # None supports COUNT(*)
    attribute: Optional[str] = None


# --------------------------------------------------
# Intent Layer
# --------------------------------------------------

class IntentCore(BaseModel):
    """
    High-level structural description of a query intent.

    Stores structural metadata independent from graph traversal
    specification, including:

    - generation regime
    - structural modifiers
    - query expansion characteristics
    """

    model_config = ConfigDict(extra="forbid")

    regime: Optional[str] = Field(
        default=None,
        description=(
            "Structural query regime identifier responsible for "
            "generating the intent specification."
        ),
    )

    # structural operators applied to the base intent
    modifiers: List[StructuralModifier] = Field(
        default_factory=list,
        description=(
            "Structural modifiers applied to the base query intent "
            "(e.g. FILTER, AGGREGATE, COUNT, ORDER_BY, LIMIT)."
        ),
    )


# --------------------------------------------------
# Schema Layer
# --------------------------------------------------

class SchemaSpec(BaseModel):
    """
    Structural graph query specification.

    Defines how the intent maps to graph traversal operations,
    including:

    - target entities
    - traversal paths
    - attribute constraints
    - aggregation behavior
    - ordering rules
    - result limits
    """

    model_config = ConfigDict(extra="forbid")

    # main entity returned by the query
    target: Dict[str, Any]

    # traversal specification connecting graph entities
    path: List[Dict[str, Any]] = Field(default_factory=list)

    # attribute constraints applied across query nodes
    filters: List[AttributeFilter] = Field(default_factory=list)

    # result ordering specification
    order_by: Optional[Dict[str, Any]] = None

    # maximum number of returned results
    limit: Optional[int] = None

    # aggregation operation specification
    aggregate: Optional[AggregateSpec] = None

    # attributes returned for matched target nodes
    return_attributes: List[str] = Field(default_factory=list)


# --------------------------------------------------
# Root Intent Specification
# --------------------------------------------------

class IntentSpec(BaseModel):
    """
    Canonical root representation of a structural graph query intent.

    Combines:

    - structural intent metadata
    - graph traversal specification
    - query constraints
    - structural modifiers

    This model serves as the central data representation shared
    across generation, validation and dataset export pipelines.
    """

    model_config = ConfigDict(
        populate_by_name=True,

        # prevent silent schema drift during experimentation
        extra="forbid"
    )

    intent: IntentCore

    # alias preserves backward compatibility with dataset schema
    schema_spec: SchemaSpec = Field(alias="schema")

    # --------------------------------------------------
    # Helper methods
    # --------------------------------------------------

    def has_filter(self) -> bool:
        """
        Check whether the intent contains filter constraints.
        """

        return StructuralModifier.FILTER in self.intent.modifiers


    def has_aggregate(self) -> bool:
        """
        Check whether the intent applies aggregation operations.
        """

        return StructuralModifier.AGGREGATE in self.intent.modifiers


    def has_order_by(self) -> bool:
        """
        Check whether the intent defines result ordering.
        """

        return StructuralModifier.ORDER_BY in self.intent.modifiers


    def has_limit(self) -> bool:
        """
        Check whether the intent constrains result size.
        """

        return StructuralModifier.LIMIT in self.intent.modifiers


    # --------------------------------------------------
    # Export helper
    # --------------------------------------------------

    def to_json(self) -> str:
        """
        Serialize the intent preserving schema alias compatibility.

        Ensures exported JSON matches the canonical dataset format:

        {
            "intent": {...},
            "schema": {...}
        }

        Returns
        -------
        str
            Serialized JSON representation of the intent.
        """

        return self.model_dump_json(by_alias=True)
