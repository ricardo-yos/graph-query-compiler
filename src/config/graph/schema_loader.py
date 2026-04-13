"""
Graph Schema Loader
===================

Loads the graph schema configuration used across the query compiler.

This module is responsible for reading the static graph structure
definition from a JSON file and preparing it for runtime usage.

Responsibilities
----------------
- Load graph schema from disk
- Ensure consistent structure for downstream validation
- Normalize specific fields (e.g., traversal patterns)

Notes
-----
- The schema is treated as a static configuration
- Any transformation here must be deterministic
- Used primarily by the SchemaValidator
"""

import json
from pathlib import Path


def load_graph_schema() -> dict:
    """
    Load and prepare the graph schema configuration.

    This function reads the graph schema from a JSON file and applies
    minimal normalization required for runtime validation.

    Specifically:
    - Converts traversal patterns into tuples for hashability
      and efficient comparison during validation

    Returns
    -------
    dict
        Fully loaded and normalized graph schema
    """
    # Resolve schema file path relative to this module
    schema_path = Path(__file__).parent / "graph_schema.json"

    # Load JSON schema from disk
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    # Convert patterns to tuples (required for set membership checks)
    schema["patterns"] = [tuple(p) for p in schema.get("patterns", [])]

    return schema
