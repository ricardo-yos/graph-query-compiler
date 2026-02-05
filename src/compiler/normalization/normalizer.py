"""
Schema Normalizer
=================

This module implements the structural normalization stage of the
semantic compilation pipeline.

Its primary responsibility is to guarantee that the Intermediate
Representation (IR) produced by the LLM is structurally consistent,
predictable, and safe for downstream processing.

This stage performs ONLY structural normalization, and does NOT apply
any semantic validation, inference, or correction.

Key responsibilities:
- Guarantee the existence of all required schema fields
- Normalize field types and container formats (lists and dicts)
- Convert null values into safe defaults
- Enforce a canonical structural representation
- Protect downstream stages from malformed LLM outputs

Pipeline:
LLM Output → SchemaNormalizer → IRValidator → CypherGenerator
"""

from typing import Dict, Any
import copy


class SchemaNormalizer:
    """
    Structural normalizer for the Intermediate Representation (IR).

    This class enforces a canonical and stable schema structure,
    ensuring that all downstream components can operate safely
    and deterministically on the IR.

    It does not perform semantic validation or interpretation — only
    structural normalization and sanitization.
    """

    DEFAULT_SCHEMA = {
        "constraints": {
            "filters": [],
            "limit": [],
            "order_by": []
        },
        "known": [],
        "path": [],
        "query_pattern": "",
        "return": {}
    }

    @classmethod
    def normalize(cls, ir: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs full structural normalization of the raw IR.

        This method ensures that:
        - All mandatory fields exist
        - All schema fields follow consistent and predictable types
        - All null values are converted into safe defaults
        - List and dictionary formats are normalized

        This step guarantees that the IR can be safely consumed by
        the validator and Cypher generator without requiring
        defensive checks.

        Notes
        -----
        This method performs ONLY structural normalization.
        No semantic validation, correction, or inference is applied.

        Parameters
        ----------
        ir : Dict[str, Any]
            Raw intermediate representation produced by the LLM.

        Returns
        -------
        Dict[str, Any]
            Structurally normalized IR, safe for validation and
            Cypher generation.
        """
        ir = copy.deepcopy(ir)

        # Ensure top-level fields
        if not isinstance(ir, dict):
            ir = {"user_intent": "retrieve", "schema": copy.deepcopy(cls.DEFAULT_SCHEMA)}
        else:
            ir.setdefault("user_intent", "retrieve")
            schema = ir.get("schema")
            if not isinstance(schema, dict):
                ir["schema"] = copy.deepcopy(cls.DEFAULT_SCHEMA)
            else:
                cls._normalize_schema(schema)

        return ir

    @classmethod
    def _normalize_schema(cls, schema: Dict[str, Any]) -> None:
        """
        Normalizes the internal schema structure.

        Ensures that all schema-level fields:
        - Exist
        - Have valid structural types
        - Follow a canonical container format

        This method applies defensive normalization to protect
        downstream stages from malformed or partially structured
        LLM outputs.
        """
        for key, default in cls.DEFAULT_SCHEMA.items():
            if key not in schema or schema[key] is None:
                schema[key] = copy.deepcopy(default)
            elif not isinstance(schema[key], type(default)):
                schema[key] = copy.deepcopy(default)

        # Normalize constraints
        constraints = schema["constraints"]
        for key in ["filters", "limit", "order_by"]:
            val = constraints.get(key)
            if val is None:
                constraints[key] = []
            elif isinstance(val, dict) and key == "filters":
                constraints[key] = [val]

        # Normalize known
        known = schema.get("known") or []
        if isinstance(known, dict):
            known = [known]
        schema["known"] = [
            k if isinstance(k, dict) else {}
            for k in known
        ]

        # Normalize path
        path = schema.get("path") or []
        schema["path"] = path if isinstance(path, list) else []

        # Normalize return
        ret = schema.get("return") or {}
        schema["return"] = ret if isinstance(ret, dict) else {}
