"""
Intent Specification Models
===========================

Canonical data models representing structured graph query intents.

These models define the contract between:
- structural generator
- natural language generation pipeline
- downstream semantic parsing tasks

Design goals:
- enforce schema consistency
- prevent silent field drift
- support deterministic dataset generation
- preserve compatibility with existing datasets
"""

from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict


# --------------------------------------------------
# Enums
# --------------------------------------------------

class PrimaryIntent(str, Enum):
    """
    High-level intent category.

    Currently only retrieval is supported, but the enum
    allows future expansion (e.g., compare, recommend).
    """

    RETRIEVE = "retrieve"


class StructuralModifier(str, Enum):
    """
    Structural operators applied to the base retrieval intent.

    These modifiers determine how the query structure is expanded.
    """

    FILTER = "filter"
    AGGREGATE = "aggregate"
    ORDER_BY = "order_by"
    LIMIT = "limit"


# --------------------------------------------------
# Filter model
# --------------------------------------------------

class AttributeFilter(BaseModel):
    """
    Represents a constraint applied to a node attribute.

    Example:
        Movie.rating > 8
        Person.birth_year >= 1990
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
    Represents an aggregation operation applied to the result set.

    Examples:
        avg(Movie.rating)
        count(*)
        max(Movie.revenue)
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
    High-level description of the user's query goal.

    Contains semantic intent information independent of graph structure.
    """

    model_config = ConfigDict(extra="forbid")

    type: PrimaryIntent = PrimaryIntent.RETRIEVE

    # structural operators applied to the base intent
    modifiers: List[StructuralModifier] = Field(default_factory=list)

    regime: Optional[str] = Field(
        default=None,

        # helps trace which generation configuration produced the intent
        description="Structural generation regime identifier"
    )


# --------------------------------------------------
# Schema Layer
# --------------------------------------------------

class SchemaSpec(BaseModel):
    """
    Graph query structure specification.

    Defines how the intent maps to graph traversal operations.
    """

    model_config = ConfigDict(extra="forbid")

    # main entity being queried
    target: Dict[str, Any]

    # traversal chain connecting entities
    path: List[Dict[str, Any]] = Field(default_factory=list)

    # attribute constraints applied to nodes
    filters: List[AttributeFilter] = Field(default_factory=list)

    # ordering criteria
    order_by: Optional[Dict[str, Any]] = None

    # maximum number of returned results
    limit: Optional[int] = None

    # aggregation specification
    aggregate: Optional[AggregateSpec] = None

    # attributes returned for each matched node
    return_attributes: List[str] = Field(default_factory=list)


# --------------------------------------------------
# Root Intent Specification
# --------------------------------------------------

class IntentSpec(BaseModel):
    """
    Root container combining semantic intent and graph schema structure.

    This is the canonical representation used across the pipeline.
    """

    model_config = ConfigDict(
        populate_by_name=True,

        # prevents silent schema drift during experimentation
        extra="forbid"
    )

    intent: IntentCore

    # alias preserves compatibility with existing dataset format
    schema_spec: SchemaSpec = Field(alias="schema")

    # --------------------------------------------------
    # Helper methods
    # --------------------------------------------------

    def has_filter(self) -> bool:
        """
        Check if the intent contains filter constraints.
        """

        return StructuralModifier.FILTER in self.intent.modifiers


    def has_aggregate(self) -> bool:
        """
        Check if the intent applies aggregation.
        """

        return StructuralModifier.AGGREGATE in self.intent.modifiers


    def has_order_by(self) -> bool:
        """
        Check if the intent defines result ordering.
        """

        return StructuralModifier.ORDER_BY in self.intent.modifiers


    def has_limit(self) -> bool:
        """
        Check if the intent limits the number of results.
        """

        return StructuralModifier.LIMIT in self.intent.modifiers


    # --------------------------------------------------
    # Export helper
    # --------------------------------------------------

    def to_json(self) -> str:
        """
        Serialize intent preserving schema alias compatibility.

        Ensures exported JSON matches the expected dataset format:
        {
            "intent": {...},
            "schema": {...}
        }
        """

        return self.model_dump_json(by_alias=True)
