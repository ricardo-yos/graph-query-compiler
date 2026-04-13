"""
Cypher Query Generator
======================

Compiles a validated Intermediate Representation (IR) into a Cypher query.

This module acts as the final stage of the semantic compilation pipeline:

    Natural Language
        → Schema (LLM)
        → Normalization
        → Validation
        → Cypher Generation

Responsibilities
----------------
- Translate schema into Cypher clauses
- Generate MATCH patterns (node or path)
- Apply filtering conditions (WHERE)
- Project attributes (RETURN)
- Apply ordering (ORDER BY)
- Apply limits (LIMIT)

Design Principles
-----------------
- Deterministic compilation (no inference)
- Assumes input is already validated
- Modular compilation pipeline
- Clear separation of query stages

Input Contract
--------------
Expected IR format:

{
    "schema": {
        "filters": [...],
        "limit": int | None,
        "order_by": dict | None,
        "path": list,
        "return_attributes": list,
        "target": {"label": str}
    }
}
"""

from typing import Dict, Any, List


class CypherGenerator:
    """
    Translates a validated schema into a Cypher query.

    This class builds the query incrementally using
    independent compilation stages.
    """

    def __init__(self, ir: Dict[str, Any]):
        """
        Initialize generator with validated IR.

        Parameters
        ----------
        ir : Dict[str, Any]
            Validated intermediate representation
        """
        self.ir = ir
        self.schema = ir["schema"]

        # Accumulators for query construction
        self.match_clauses: List[str] = []
        self.where_clauses: List[str] = []
        self.return_clause: str = ""
        self.order_clause: str = ""
        self.limit_clause: str = ""

        # Maps node labels to Cypher variables
        # Example: {"Place": "p"}
        self.alias_map: Dict[str, str] = {}

    # -------------------------------------------------
    def generate(self) -> str:
        """
        Generate the full Cypher query.

        Returns
        -------
        str
            Fully compiled Cypher query
        """
        self._compile_match()
        self._compile_filters()
        self._compile_return()
        self._compile_order_by()
        self._compile_limit()

        parts = [
            *self.match_clauses,
            self._where_block(),
            self.return_clause,
            self.order_clause,
            self.limit_clause,
        ]

        return "\n".join(p for p in parts if p.strip())

    # -------------------------------------------------
    def _compile_match(self):
        """
        Compile MATCH clause.

        Decides between:
        - node pattern
        - path traversal
        """
        path = self.schema.get("path", [])

        if not path:
            self._compile_node_match()
        else:
            self._compile_path_match()

    # -------------------------------------------------
    def _compile_node_match(self):
        """
        Compile single-node MATCH.

        Example:
        MATCH (p:Place)
        """
        label = self.schema["target"]["label"]

        # Alias = first letter of label
        var = label[0].lower()
        self.alias_map[label] = var

        self.match_clauses.append(f"MATCH ({var}:{label})")

    # -------------------------------------------------
    def _compile_path_match(self):
        """
        Compile multi-hop traversal MATCH.

        Example:
        MATCH (p:Place)-[:HAS_REVIEW]->(r:Review)
        """
        path = self.schema.get("path", [])
        parts = []

        for step in path:
            from_label, rel, to_label = step

            # Reuse alias if already created
            from_var = self.alias_map.get(from_label)
            if not from_var:
                from_var = from_label[0].lower()
                self.alias_map[from_label] = from_var

            to_var = self.alias_map.get(to_label)
            if not to_var:
                to_var = to_label[0].lower()
                self.alias_map[to_label] = to_var

            parts.append(
                f"({from_var}:{from_label})-[:{rel}]->({to_var}:{to_label})"
            )

        self.match_clauses.append("MATCH " + "".join(parts))

    # -------------------------------------------------
    def _compile_filters(self):
        """
        Compile WHERE conditions.

        Applies all filter constraints as AND conditions.
        """
        for f in self.schema.get("filters", []):
            # Fallback to target label if not specified
            label = f.get("node_label") or self.schema["target"]["label"]
            attr = f["attribute"]
            op = f["operator"]

            # Resolve value from available fields
            value = (
                f.get("value_str")
                if f.get("value_str") is not None
                else f.get("value_int")
                if f.get("value_int") is not None
                else f.get("value_float")
            )

            var = self.alias_map[label]

            self.where_clauses.append(
                self._format_condition(var, attr, op, value)
            )

    # -------------------------------------------------
    def _compile_return(self):
        """
        Compile RETURN clause.

        If no attributes specified, return full node.
        """
        label = self.schema["target"]["label"]
        attrs = self.schema.get("return_attributes", [])

        var = self.alias_map[label]

        if not attrs:
            self.return_clause = f"RETURN {var}"
            return

        projections = [f"{var}.{a} AS {a}" for a in attrs]
        self.return_clause = f"RETURN {', '.join(projections)}"

    # -------------------------------------------------
    def _compile_order_by(self):
        """
        Compile ORDER BY clause.
        """
        order = self.schema.get("order_by")

        if not order:
            return

        label = order.get("node_label") or self.schema["target"]["label"]
        attr = order["attribute"]
        direction = order.get("direction", "ASC")

        var = self.alias_map[label]

        self.order_clause = f"ORDER BY {var}.{attr} {direction}"

    # -------------------------------------------------
    def _compile_limit(self):
        """
        Compile LIMIT clause.
        """
        limit = self.schema.get("limit")

        if isinstance(limit, int):
            self.limit_clause = f"LIMIT {limit}"

    # -------------------------------------------------
    def _where_block(self):
        """
        Combine all WHERE conditions into a single clause.
        """
        if not self.where_clauses:
            return ""
        return "WHERE " + " AND ".join(self.where_clauses)

    # -------------------------------------------------
    @staticmethod
    def _format_condition(var: str, attr: str, op: str, value: Any) -> str:
        """
        Format a single WHERE condition.

        Handles string quoting automatically.
        """
        if isinstance(value, str):
            return f"{var}.{attr} {op} '{value}'"
        return f"{var}.{attr} {op} {value}"


# -------------------------------------------------
def compile_to_cypher(ir: Dict[str, Any]) -> str:
    """
    High-level helper to compile IR into Cypher.

    Parameters
    ----------
    ir : Dict[str, Any]
        Validated intermediate representation

    Returns
    -------
    str
        Cypher query
    """
    return CypherGenerator(ir).generate()
