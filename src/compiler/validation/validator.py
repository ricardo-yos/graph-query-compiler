"""
Schema Validator
================

Performs semantic validation over a normalized Intermediate Representation (IR)
generated from a natural language query.

This module ensures that the schema is valid with respect to the graph structure,
including node labels, properties, operators, and traversal patterns.

Pipeline Position
-----------------
LLM Output → SchemaNormalizer → SchemaValidator → CypherGenerator

Responsibilities
----------------
- Validate target node existence
- Validate filters (attributes, operators, value types)
- Validate return attributes
- Validate graph traversal paths

Design Principles
-----------------
- Assumes input is already structurally normalized
- Does NOT perform correction or inference
- Fails fast on invalid schemas
- Uses external graph schema (config-driven)

Input Contract
--------------
Expected IR format:

{
    "user_intent": "retrieve",
    "schema": {
        "filters": [...],
        "limit": ...,
        "order_by": ...,
        "path": [...],
        "return_attributes": [...],
        "target": {"label": "..."}
    }
}
"""

from typing import Dict, Any
from src.config.graph.schema_loader import load_graph_schema


class SchemaValidationError(Exception):
    """
    Exception raised when the schema is semantically invalid.
    """
    pass


class SchemaValidator:
    """
    Validates a normalized schema against the graph definition.

    This class enforces semantic correctness by checking the schema
    against a predefined graph structure loaded from configuration.
    """

    _GRAPH_SCHEMA = None

    # Allowed comparison operators for filters
    VALID_OPERATORS = {"=", "!=", ">", "<", ">=", "<="}

    # Defines which input value types are compatible with schema types
    TYPE_COMPATIBILITY = {
        "float": ["float", "int"],   # int allowed for numeric comparisons
        "int": ["int"],
        "string": ["string"],
        "bool": ["bool"]
    }

    # --------------------------------------------------
    # Schema loader (lazy initialization)
    # --------------------------------------------------

    @classmethod
    def get_schema(cls) -> Dict[str, Any]:
        """
        Lazily load and cache the graph schema.

        Returns
        -------
        dict
            Graph schema definition loaded from config
        """
        if cls._GRAPH_SCHEMA is None:
            cls._GRAPH_SCHEMA = load_graph_schema()
        return cls._GRAPH_SCHEMA

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    @classmethod
    def validate(cls, ir: Dict[str, Any]) -> None:
        """
        Validate a normalized IR.

        This method orchestrates all validation steps.

        Parameters
        ----------
        ir : Dict[str, Any]
            Normalized intermediate representation

        Raises
        ------
        SchemaValidationError
            If any semantic rule is violated
        """
        if "schema" not in ir:
            raise SchemaValidationError("Missing 'schema' in IR")

        schema = ir["schema"]

        cls._validate_target(schema)
        cls._validate_filters(schema)
        cls._validate_return(schema)
        cls._validate_path(schema)

    # --------------------------------------------------
    # Core validations
    # --------------------------------------------------

    @classmethod
    def _validate_target(cls, schema: Dict[str, Any]) -> None:
        """
        Validate the target node label.

        Ensures the query is anchored to a valid node type.
        """
        label = schema.get("target", {}).get("label")

        if not label:
            raise SchemaValidationError("Missing target label")

        graph = cls.get_schema()

        if label not in graph["nodes"]:
            raise SchemaValidationError(f"Invalid target label: {label}")

    @classmethod
    def _validate_filters(cls, schema: Dict[str, Any]) -> None:
        """
        Validate all filter conditions.
        """
        for f in schema.get("filters", []):
            cls._validate_filter(f, schema)

    @classmethod
    def _validate_filter(cls, f: Dict[str, Any], schema: Dict[str, Any]) -> None:
        """
        Validate a single filter condition.

        Notes
        -----
        If node_label is missing, fallback to target label.
        """
        if not isinstance(f, dict):
            raise SchemaValidationError(f"Invalid filter format: {f}")

        graph = cls.get_schema()

        # Fallback: use target label if filter does not specify one
        label = f.get("node_label") or schema.get("target", {}).get("label")
        attr = f.get("attribute")
        op = f.get("operator")

        # Validate label
        if not label:
            raise SchemaValidationError(f"Missing node label in filter: {f}")

        if label not in graph["nodes"]:
            raise SchemaValidationError(f"Invalid node label: {label}")

        # Validate attribute existence
        node_props = graph["nodes"][label]["properties"]

        if attr not in node_props:
            raise SchemaValidationError(
                f"Invalid attribute '{attr}' for node '{label}'"
            )

        # Validate operator
        if op not in cls.VALID_OPERATORS:
            raise SchemaValidationError(f"Invalid operator: {op}")

        # Validate value type consistency
        cls._validate_value_type(label, attr, f)

    # --------------------------------------------------
    # Value validation
    # --------------------------------------------------

    @classmethod
    def _validate_value_type(cls, label: str, attr: str, f: Dict[str, Any]) -> None:
        """
        Validate that the filter value matches the expected property type.

        Rules
        -----
        - Exactly one value field must be provided
        - Type must be compatible with schema definition
        """
        graph = cls.get_schema()

        expected_type = graph["nodes"][label]["properties"][attr]

        # Extract which value field was provided
        value_fields = {
            "float": f.get("value_float"),
            "int": f.get("value_int"),
            "string": f.get("value_str"),
            "bool": f.get("value_bool"),
        }

        provided = [k for k, v in value_fields.items() if v is not None]

        if len(provided) != 1:
            raise SchemaValidationError(
                f"Filter must have exactly one value field: {f}"
            )

        actual_type = provided[0]

        # Get allowed types for this attribute
        allowed_types = cls.TYPE_COMPATIBILITY.get(expected_type)

        if not allowed_types:
            raise SchemaValidationError(
                f"Unsupported property type '{expected_type}' for '{attr}'"
            )

        if actual_type not in allowed_types:
            raise SchemaValidationError(
                f"Expected {expected_type} for '{attr}', got {actual_type}"
            )

    # --------------------------------------------------
    # Return validation
    # --------------------------------------------------

    @classmethod
    def _validate_return(cls, schema: Dict[str, Any]) -> None:
        """
        Validate return attributes.

        Ensures all requested attributes exist in the target node.
        """
        label = schema.get("target", {}).get("label")

        graph = cls.get_schema()

        if label not in graph["nodes"]:
            raise SchemaValidationError(f"Invalid target label: {label}")

        node_props = graph["nodes"][label]["properties"]

        for attr in schema.get("return_attributes", []):
            if attr not in node_props:
                raise SchemaValidationError(
                    f"Invalid return attribute '{attr}' for '{label}'"
                )

    # --------------------------------------------------
    # Path validation
    # --------------------------------------------------

    @classmethod
    def _validate_path(cls, schema: Dict[str, Any]) -> None:
        """
        Validate graph traversal paths.

        Each step must match an allowed pattern defined in the graph schema.

        Expected format:
        [
            ("Place", "HAS_REVIEW", "Review"),
            ...
        ]
        """
        path = schema.get("path", [])

        if not path:
            return

        graph = cls.get_schema()
        valid_patterns = set(tuple(p) for p in graph.get("patterns", []))

        for step in path:
            if not isinstance(step, (list, tuple)) or len(step) != 3:
                raise SchemaValidationError(f"Invalid path step format: {step}")

            step_tuple = tuple(step)

            if step_tuple not in valid_patterns:
                raise SchemaValidationError(
                    f"Invalid graph traversal: {step_tuple}"
                )
