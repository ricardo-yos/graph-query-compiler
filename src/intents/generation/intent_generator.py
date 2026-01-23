"""
Structured Intent Generation Module
===================================

Component responsible for programmatically generating structured
query intents derived from a graph schema.

This module defines the `IntentGenerator` class, which systematically
derives a controlled and diverse set of intent specifications based on:

- Available node labels and their properties
- Allowed relationship traversal patterns
- Operator policies associated with each attribute

The generated intents are fully structured and deterministic, and are
designed to be consumed by downstream pipelines responsible for:

- Intent normalization and validation
- Scoring and filtering
- Natural language question generation
- Dataset construction for LLM fine-tuning and evaluation

This module focuses strictly on *semantic structure*, not language.
No natural language is produced at this stage.

Important Characteristics
-------------------------
- Deterministic generation (fixed random seed)
- Controlled combinatorial expansion
- Explicit coverage of common graph query patterns:
  - Retrieval
  - Filtering
  - Ranking
  - Multi-hop traversal
- Schema-driven and domain-agnostic
- Produces machine-readable intent specifications only

Notes
-----
- The output of this module is considered *canonical* and should be
  validated and possibly filtered before language generation.
- Linguistic variation is intentionally handled in later stages.
"""

import json
import random
from typing import List, Optional

from .intent_models import IntentSpec, AttributeFilter
from .operator_policy import get_operators


# --------------------------------------------------
# Intent generator
# --------------------------------------------------

class IntentGenerator:
    """
    Generator for structured intent specifications derived
    from a graph schema.

    The generator produces intents across multiple semantic categories
    (retrieve, filter, rank), while enforcing controlled combinatorial
    expansion and deterministic behavior.

    Parameters
    ----------
    schema : dict
        Graph schema definition containing:
        - nodes and their properties
        - allowed relationship traversal patterns
    max_variants : int, optional
        Maximum number of variants for certain intent classes.
    max_two_hop_intents : int, optional
        Upper bound for generated two-hop traversal intents.
    max_two_hop_filtered_intents : int, optional
        Upper bound for two-hop intents with attribute filters.
    """

    def __init__(
        self,
        schema: dict,
        max_variants: int = 3,
        max_two_hop_intents: int = 10,
        max_two_hop_filtered_intents: int = 5,
    ):
        # Deterministic generation
        random.seed(42)

        self.nodes = schema["nodes"]
        self.allowed_patterns = schema["allowed_patterns"]

        self.max_variants = max_variants
        self.max_two_hop_intents = max_two_hop_intents
        self.max_two_hop_filtered_intents = max_two_hop_filtered_intents

    # --------------------------------------------------
    # Helper utilities
    # --------------------------------------------------

    def _default_value_for(self, attribute: str, label: Optional[str] = None) -> object:
        """
        Resolve a semantically plausible default value for an attribute.

        Parameters
        ----------
        attribute : str
            Attribute name.
        label : str, optional
            Node label associated with the attribute.

        Returns
        -------
        Any
            A deterministic, semantically valid placeholder value.
        """

        # -------------------------------
        # String identifiers
        # -------------------------------
        if attribute in {"name", "place_name", "road_name", "neighborhood_name"}:
            if label:
                label = label.lower()

                if label == "neighborhood":
                    return "Bairro Alfa"
                if label == "road":
                    return "Rua Central"
                if label == "place":
                    return "Pet Shop Alfa"

            return "Nome Exemplo"

        # -------------------------------
        # Ratings / scores
        # -------------------------------
        if attribute == "rating":
            return 4

        # -------------------------------
        # Counts
        # -------------------------------
        if attribute in {"num_reviews", "reviews_count"}:
            return 10

        # -------------------------------
        # Neighborhood numeric attributes
        # -------------------------------
        if attribute == "area_km2":
            return 1_000_000

        if attribute == "average_monthly_income":
            return 1_000

        # -------------------------------
        # Road attributes
        # -------------------------------
        if attribute == "maxspeed":
            return 60

        if attribute == "highway":
            return 0

        # -------------------------------
        # Place attributes
        # -------------------------------
        if attribute == "type":
            return "pet_shop"

        # -------------------------------
        # Generic fallback
        # -------------------------------
        return 1

    def _known_input_with_value(self, label: str, attribute: str, operator: str):
        """
        Build a known-input constraint with a valid placeholder value.

        Returns
        -------
        dict
            Known-input constraint dictionary.
        """
        return {
            "label": label,
            "attribute": attribute,
            "operator": operator,
            "value": self._default_value_for(attribute, label),
        }

    # --------------------------------------------------
    # 1. Retrieve intents (0-hop)
    # --------------------------------------------------

    def generate_list_intents(self) -> List[IntentSpec]:
        intents = []

        for label in self.nodes:
            intents.append(
                IntentSpec(
                    intent_type="retrieve",
                    known_inputs=[],
                    target={"label": label},
                    path=[],
                    filters=[],
                )
            )

        return intents

    # --------------------------------------------------
    # 2. Lookup by known attribute (0-hop)
    # --------------------------------------------------

    def generate_lookup_by_known_attribute(self) -> List[IntentSpec]:
        intents = []

        for label, node_data in self.nodes.items():
            for prop in node_data.get("properties", []):
                intents.append(
                    IntentSpec(
                        intent_type="retrieve",
                        known_inputs=[
                            self._known_input_with_value(label, prop, "=")
                        ],
                        target={"label": label},
                        path=[],
                        filters=[],
                    )
                )

        return intents

    # --------------------------------------------------
    # 3. Lookup by relationship (1-hop)
    # --------------------------------------------------

    def generate_lookup_by_relationship(self) -> List[IntentSpec]:
        intents = []

        for pattern in self.allowed_patterns:
            intents.append(
                IntentSpec(
                    intent_type="retrieve",
                    known_inputs=[
                        self._known_input_with_value(
                            pattern["start"], "name", "="
                        )
                    ],
                    path=[
                        {
                            "from": pattern["start"],
                            "rel": pattern["relationship"],
                            "to": pattern["end"],
                        }
                    ],
                    target={"label": pattern["end"]},
                    filters=[],
                )
            )

        return intents

    # --------------------------------------------------
    # 4. Lookup by relationship (2-hop)
    # --------------------------------------------------

    def generate_lookup_by_two_hop_relationship(self) -> List[IntentSpec]:
        intents = []

        for p1 in self.allowed_patterns:
            for p2 in self.allowed_patterns:
                if p1["end"] != p2["start"]:
                    continue
                if p1["start"] == p2["end"]:
                    continue

                intents.append(
                    IntentSpec(
                        intent_type="retrieve",
                        known_inputs=[
                            self._known_input_with_value(
                                p1["start"], "name", "="
                            )
                        ],
                        path=[
                            {
                                "from": p1["start"],
                                "rel": p1["relationship"],
                                "to": p1["end"],
                            },
                            {
                                "from": p2["start"],
                                "rel": p2["relationship"],
                                "to": p2["end"],
                            },
                        ],
                        target={"label": p2["end"]},
                        filters=[],
                    )
                )

        random.shuffle(intents)
        return intents[: self.max_two_hop_intents]

    # --------------------------------------------------
    # 5. Two-hop with filters
    # --------------------------------------------------

    def generate_two_hop_with_filters(self) -> List[IntentSpec]:
        base_intents = self.generate_lookup_by_two_hop_relationship()
        filtered_intents = []

        for intent in base_intents:
            target_label = intent.target["label"]
            target_props = self.nodes.get(target_label, {}).get("properties", [])

            for prop in target_props:
                for operator in get_operators(prop):
                    filtered_intents.append(
                        IntentSpec(
                            intent_type="filter",
                            known_inputs=intent.known_inputs,
                            path=intent.path,
                            target=intent.target,
                            filters=[
                                AttributeFilter(
                                    node_label=target_label,
                                    attribute=prop,
                                    operator=operator,
                                    value=self._default_value_for(prop, target_label),
                                )
                            ],
                        )
                    )

        random.shuffle(filtered_intents)
        return filtered_intents[: self.max_two_hop_filtered_intents]

    # --------------------------------------------------
    # 6. Filter by attribute (0-hop)
    # --------------------------------------------------

    def generate_filter_by_attribute(self) -> List[IntentSpec]:
        intents = []

        for label, node_data in self.nodes.items():
            for prop in node_data.get("properties", []):
                for operator in get_operators(prop):
                    intents.append(
                        IntentSpec(
                            intent_type="filter",
                            known_inputs=[],
                            target={"label": label},
                            path=[],
                            filters=[
                                AttributeFilter(
                                    node_label=label,
                                    attribute=prop,
                                    operator=operator,
                                    value=self._default_value_for(prop, label),
                                )
                            ],
                        )
                    )

        return intents

    # --------------------------------------------------
    # 7. Multi-filter (0-hop)
    # --------------------------------------------------

    def generate_multi_filter_intents(self) -> List[IntentSpec]:
        intents = []

        for label, node_data in self.nodes.items():
            props = node_data.get("properties", [])
            if len(props) < 2:
                continue

            for p1 in props:
                for p2 in props:
                    if p1 == p2:
                        continue

                    intents.append(
                        IntentSpec(
                            intent_type="filter",
                            known_inputs=[],
                            target={"label": label},
                            path=[],
                            filters=[
                                AttributeFilter(
                                    node_label=label,
                                    attribute=p1,
                                    operator=get_operators(p1)[0],
                                    value=self._default_value_for(p1, label),
                                ),
                                AttributeFilter(
                                    node_label=label,
                                    attribute=p2,
                                    operator=get_operators(p2)[0],
                                    value=self._default_value_for(p2, label),
                                ),
                            ],
                        )
                    )

        return intents

    # --------------------------------------------------
    # 8. Filter + ordering
    # --------------------------------------------------

    def generate_filter_with_ordering(self) -> List[IntentSpec]:
        intents = []

        for label, node_data in self.nodes.items():
            if "rating" not in node_data.get("properties", []):
                continue

            intents.append(
                IntentSpec(
                    intent_type="filter",
                    known_inputs=[],
                    target={"label": label},
                    path=[],
                    filters=[],
                    order_by={
                        "label": label,
                        "attribute": "rating",
                        "direction": "desc",
                    },
                    limit=5,
                )
            )

        return intents

    # --------------------------------------------------
    # 9. Rank with known input
    # --------------------------------------------------

    def generate_rank_with_known(self) -> List[IntentSpec]:
        intents = []

        for pattern in self.allowed_patterns:
            intents.append(
                IntentSpec(
                    intent_type="rank",
                    known_inputs=[
                        self._known_input_with_value(
                            pattern["start"], "name", "="
                        )
                    ],
                    path=[
                        {
                            "from": pattern["start"],
                            "rel": pattern["relationship"],
                            "to": pattern["end"],
                        }
                    ],
                    target={"label": pattern["end"]},
                    filters=[],
                    order_by={
                        "label": pattern["end"],
                        "attribute": "rating",
                        "direction": "desc",
                    },
                    limit=5,
                )
            )

        return intents

    # --------------------------------------------------
    # 10. Range filters
    # --------------------------------------------------

    def generate_range_filter_intents(self) -> List[IntentSpec]:
        intents = []

        for label, node_data in self.nodes.items():
            for prop in node_data.get("properties", []):
                if prop in {"rating", "num_reviews", "area_km2"}:
                    intents.append(
                        IntentSpec(
                            intent_type="filter",
                            known_inputs=[],
                            target={"label": label},
                            path=[],
                            filters=[
                                AttributeFilter(
                                    node_label=label,
                                    attribute=prop,
                                    operator=">",
                                    value=2,
                                ),
                                AttributeFilter(
                                    node_label=label,
                                    attribute=prop,
                                    operator="<",
                                    value=5,
                                ),
                            ],
                        )
                    )

        return intents

    # --------------------------------------------------
    # 11. Has-relationship intents
    # --------------------------------------------------

    def generate_has_relationship_intents(self) -> List[IntentSpec]:
        intents = []

        for pattern in self.allowed_patterns:
            intents.append(
                IntentSpec(
                    intent_type="filter",
                    known_inputs=[],
                    path=[
                        {
                            "from": pattern["start"],
                            "rel": pattern["relationship"],
                            "to": pattern["end"],
                        }
                    ],
                    target={"label": pattern["start"]},
                    filters=[],
                )
            )

        return intents

    # --------------------------------------------------
    # 12. Indirect rank intents
    # --------------------------------------------------

    def generate_indirect_rank_intents(self) -> List[IntentSpec]:
        intents = []

        for pattern in self.allowed_patterns:
            end_props = self.nodes.get(pattern["end"], {}).get("properties", [])
            if "rating" not in end_props:
                continue

            intents.append(
                IntentSpec(
                    intent_type="rank",
                    known_inputs=[],
                    path=[
                        {
                            "from": pattern["start"],
                            "rel": pattern["relationship"],
                            "to": pattern["end"],
                        }
                    ],
                    target={"label": pattern["start"]},
                    filters=[],
                    order_by={
                        "label": pattern["end"],
                        "attribute": "rating",
                        "direction": "desc",
                    },
                    limit=5,
                )
            )

        return intents

    # --------------------------------------------------
    # Aggregate generation
    # --------------------------------------------------

    def generate_all(self) -> List[IntentSpec]:
        """
        Generate all supported intent specifications.
        """
        return (
            self.generate_list_intents()
            + self.generate_lookup_by_known_attribute()
            + self.generate_lookup_by_relationship()
            + self.generate_lookup_by_two_hop_relationship()
            + self.generate_two_hop_with_filters()
            + self.generate_filter_by_attribute()
            + self.generate_multi_filter_intents()
            + self.generate_filter_with_ordering()
            + self.generate_rank_with_known()
            + self.generate_range_filter_intents()
            + self.generate_has_relationship_intents()
            + self.generate_indirect_rank_intents()
        )

    # --------------------------------------------------
    # Dataset serialization
    # --------------------------------------------------

    def to_dataset_intent(self, intent: IntentSpec) -> dict:
        """
        Convert an IntentSpec into the normalized dataset
        representation used by downstream pipelines.
        """

        filters = []
        for f in intent.filters:
            value = f.value
            filters.append(
                {
                    "label": f.node_label,
                    "attribute": f.attribute,
                    "operator": f.operator,
                    "value_str": value if isinstance(value, str) else None,
                    "value_int": value if isinstance(value, int) else None,
                }
            )

        known = []
        for k in intent.known_inputs or []:
            value = k["value"]
            known.append(
                {
                    "label": k["label"],
                    "attribute": k["attribute"],
                    "operator": k["operator"],
                    "value_str": value if isinstance(value, str) else None,
                    "value_int": value if isinstance(value, int) else None,
                }
            )

        return {
            "user_intent": intent.intent_type,
            "query_pattern": "path" if intent.path else "node",
            "path": intent.path,
            "constraints": {
                "filters": filters,
                "order_by": [intent.order_by] if intent.order_by else [],
                "limit": [intent.limit] if intent.limit else [],
            },
            "known": known,
            "return": {
                "label": intent.target["label"],
                "attributes": (
                    ["count"]
                    if intent.intent_type == "count"
                    else (
                        ["name", intent.order_by["attribute"]]
                        if intent.intent_type == "rank"
                        else ["name"]
                    )
                ),
            },
        }

    def generate_dataset(self) -> List[dict]:
        """
        Generate the full normalized intent dataset.
        """
        return [self.to_dataset_intent(i) for i in self.generate_all()]

    def save_jsonl(self, path: str) -> None:
        """
        Persist the generated dataset to a JSONL file.
        """
        with open(path, "w", encoding="utf-8") as f:
            for item in self.generate_dataset():
                f.write(json.dumps(item, ensure_ascii=False))
                f.write("\n")
