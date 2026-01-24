"""
Schema Loader
=============

Utility module responsible for loading graph schema definitions
from disk.

This module provides a minimal and explicit interface to read
a JSON-based graph schema file and return its contents as a
Python dictionary. The loaded schema is typically consumed by
intent generation, validation, and query construction pipelines.

Dependencies
------------
- json
- pathlib

Usage
-----
Import and call the loader function with a schema path:

    from schema_loader import load_schema
    schema = load_schema("data/schema/graph_schema.json")

Notes
-----
- The schema file must exist and be a valid JSON document.
- Structural validation is expected to occur downstream.
"""

import json
from pathlib import Path
from typing import Dict, Any


# --------------------------------------------------
# Schema loading utility
# --------------------------------------------------

def load_schema(schema_path: str) -> Dict[str, Any]:
    """
    Load a graph schema from a JSON file.

    Parameters
    ----------
    schema_path : str
        Path to the JSON schema file.

    Returns
    -------
    Dict[str, Any]
        Parsed schema represented as a Python dictionary.

    Raises
    ------
    FileNotFoundError
        If the schema file does not exist.

    Notes
    -----
    - This function performs minimal validation.
    - Schema structure and semantic correctness are
      validated by downstream components.
    """
    path = Path(schema_path)

    if not path.exists():
        raise FileNotFoundError(f"Schema not found: {schema_path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)
