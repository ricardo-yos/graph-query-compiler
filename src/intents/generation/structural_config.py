"""
Structural Generation Configuration
===================================

Configuration schema controlling combinatorial expansion
of structured graph query intents.

This model defines constraints that regulate:

- graph traversal depth
- filter combinatorics
- aggregation availability
- ordering and limit operators
- structural explosion control
- regime balancing strategies
- semantic distribution balancing

Design goals:
- deterministic generation
- bounded combinatorial explosion
- reproducible dataset composition
- configurable structural diversity
"""

from typing import Optional, List
from pydantic import BaseModel, Field, field_validator


class StructuralGenerationConfig(BaseModel):
    """
    Parameters controlling structural intent generation.

    The configuration balances structural diversity against
    combinatorial explosion, enabling scalable dataset creation.
    """

    # ----------------------------------------------------------
    # Pydantic configuration
    # ----------------------------------------------------------

    # forbid unknown fields to prevent silent config drift
    model_config = {
        "extra": "forbid"
    }

    # ==========================================================
    # Regime metadata
    # ==========================================================

    regime_name: Optional[str] = Field(
        default=None,

        # allows tracking dataset composition across regimes
        description="Identifier of the structural generation regime."
    )

    # ==========================================================
    # Core compositional controls
    # ==========================================================

    max_depth: int = Field(
        default=1,

        # controls maximum path length in graph traversal
        description="Maximum traversal depth starting from root node."
    )

    # ==========================================================
    # Filter controls
    # ==========================================================

    max_filters_per_node: int = Field(
        default=2,

        # limits filter combinatorial growth
        description="Maximum number of filters applied per node."
    )

    allow_multiple_filters: bool = Field(
        default=True,

        # enables combinations of attribute constraints
        description="Allow multiple filters on the same node."
    )

    max_operators_per_attribute: int = Field(
        default=2,

        # sampling avoids exponential operator combinations
        description="Maximum operators sampled per attribute."
    )

    # ==========================================================
    # Aggregation controls
    # ==========================================================

    allow_aggregation: bool = Field(
        default=True,

        # enables aggregate intent generation
        description="Allow aggregate query variants."
    )

    # ==========================================================
    # Order By controls
    # ==========================================================

    allow_order_by: bool = Field(
        default=True,

        # enables ranking-based queries
        description="Allow ordering of results."
    )

    order_by_directions: List[str] = Field(
        default_factory=lambda: ["asc", "desc"],

        # supports ranking diversity
        description="Allowed ordering directions."
    )

    # ==========================================================
    # Limit controls
    # ==========================================================

    allow_limit: bool = Field(
        default=True,

        # enables top-k style queries
        description="Allow result limiting."
    )

    limit_values: List[int] = Field(
        default_factory=lambda: [5, 10, 20],

        # common top-k values used in natural questions
        description="Allowed LIMIT values."
    )

    # ==========================================================
    # Explosion control
    # ==========================================================

    deduplicate_structures: bool = Field(
        default=True,

        # avoids repeated schemas across regimes
        description="Remove structurally duplicate intents."
    )

    # ==========================================================
    # Regime balancing
    # ==========================================================

    enforce_regime_balance: bool = Field(
        default=False,

        # ensures comparable dataset size across regimes
        description="Force balanced sampling across regimes."
    )

    target_per_regime: Optional[int] = Field(
        default=None,

        # useful when combining multiple structural policies
        description="Target number of intents per regime."
    )

    # ==========================================================
    # Semantic bucket balancing
    # ==========================================================

    enable_semantic_balance: bool = Field(
        default=False,

        # ensures distribution across semantic categories
        description="Balance dataset across semantic buckets."
    )

    min_per_bucket: int = Field(
        default=20,

        # prevents underrepresented semantic patterns
        description="Minimum intents per semantic bucket."
    )

    max_per_bucket: Optional[int] = Field(
        default=None,

        # prevents dominance of highly combinatorial buckets
        description="Maximum intents per semantic bucket."
    )

    # ==========================================================
    # Validators
    # ==========================================================

    @field_validator("max_depth")
    @classmethod
    def validate_depth(cls, v):
        """
        Depth must be non-negative.

        depth=0 generates single-node intents.
        """

        if v < 0:
            raise ValueError("max_depth must be >= 0")

        return v


    @field_validator("max_filters_per_node")
    @classmethod
    def validate_filters(cls, v):
        """
        Filter count must be non-negative.
        """

        if v < 0:
            raise ValueError("max_filters_per_node must be >= 0")

        return v


    @field_validator("max_operators_per_attribute")
    @classmethod
    def validate_operators(cls, v):
        """
        At least one operator must be available per attribute.
        """

        if v <= 0:
            raise ValueError("max_operators_per_attribute must be > 0")

        return v


    @field_validator("limit_values")
    @classmethod
    def validate_limits(cls, v):
        """
        LIMIT values must be positive integers.
        """

        if any(value <= 0 for value in v):
            raise ValueError("All limit values must be > 0")

        return v


    @field_validator("order_by_directions")
    @classmethod
    def validate_order_directions(cls, v):
        """
        Ordering direction must follow SQL convention.
        """

        allowed = {"asc", "desc"}

        if any(direction not in allowed for direction in v):

            raise ValueError(
                "order_by_directions must contain only 'asc' or 'desc'"
            )

        return v


    @field_validator("target_per_regime")
    @classmethod
    def validate_target_regime(cls, v):
        """
        Regime sampling target must be positive if provided.
        """

        if v is not None and v <= 0:
            raise ValueError("target_per_regime must be > 0")

        return v


    @field_validator("min_per_bucket")
    @classmethod
    def validate_min_bucket(cls, v):
        """
        Minimum semantic bucket size must be positive.
        """

        if v <= 0:
            raise ValueError("min_per_bucket must be > 0")

        return v


    @field_validator("max_per_bucket")
    @classmethod
    def validate_max_bucket(cls, v):
        """
        Maximum semantic bucket size must be positive if defined.
        """

        if v is not None and v <= 0:
            raise ValueError("max_per_bucket must be > 0")

        return v
