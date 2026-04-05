"""
Graph Schema Adapter
====================

Thin adapter that converts a raw graph_schema.json dictionary
into a typed object interface consumable by the structural generator.

Responsibilities:
- expose node labels
- expose node attributes
- expose allowed traversal relationships
- hide raw JSON structure from downstream components

This abstraction decouples schema format from generator logic,
allowing schema evolution without modifying generation code.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class Relationship:
    """
    Represents a directed relationship in the graph schema.

    Attributes
    ----------
    type : str
        Relationship type (edge label).

    target : str
        Target node label reachable through this relationship.
    """
    type: str
    target: str


class GraphSchemaAdapter:
    """
    Adapter that exposes graph schema information through a
    stable interface used by the structural generator.

    Expected schema format:
    {
        "nodes": {
            "Movie": {
                "properties": [...]
            }
        },
        "allowed_patterns": [
            {
                "start": "Movie",
                "relationship": "HAS_GENRE",
                "end": "Genre"
            }
        ]
    }
    """

    def __init__(self, schema_dict: dict):
        """
        Initialize adapter with raw schema dictionary.

        Parameters
        ----------
        schema_dict : dict
            Parsed graph_schema.json content.
        """

        self.schema = schema_dict

        # cached list of labels avoids repeated dictionary traversal
        self.labels = list(self.schema["nodes"].keys())

        # traversal constraints used by path generator
        self.allowed_patterns = self.schema.get("allowed_patterns", [])

    # ==========================================================
    # Node attributes
    # ==========================================================

    def get_attributes(self, label: str) -> List[str]:
        """
        Return list of attributes available for a given node label.

        Parameters
        ----------
        label : str
            Node label.

        Returns
        -------
        List[str]
            Attribute names that can be used for filtering,
            projection, aggregation, or ordering.
        """

        return self.schema["nodes"][label]["properties"]

    # ==========================================================
    # Outgoing relationships
    # ==========================================================

    def get_outgoing(self, label: str) -> List[Relationship]:
        """
        Return relationships that can be traversed starting
        from a given node label.

        Parameters
        ----------
        label : str
            Source node label.

        Returns
        -------
        List[Relationship]
            Relationships allowed by the schema traversal policy.

        Notes
        -----
        The adapter enforces traversal constraints defined in
        allowed_patterns, preventing the generator from producing
        structurally invalid paths.
        """

        outgoing = []

        for pattern in self.allowed_patterns:

            if pattern["start"] == label:

                outgoing.append(
                    Relationship(
                        type=pattern["relationship"],
                        target=pattern["end"]
                    )
                )

        return outgoing
