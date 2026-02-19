"""
Entity Type Resolver
====================

This module performs semantic normalization of entity "type" filters
within a compiled schema representation.

Its responsibility is to resolve potentially noisy, misspelled, or
semantically similar type values into a controlled set of valid types.

Input
-----
A compiled schema dictionary containing constraint filters, e.g.:

{
    "constraints": {
        "filters": [
            {"attribute": "type", "label": "Place", "value_str": "..."}
        ]
    }
}

Output
------
A structured dictionary containing:

- original_schema : Unmodified input schema
- resolved_schema : Corrected schema copy
- analysis        : Resolution trace with strategy and confidence score

Resolution Strategy
-------------------
1. Deterministic exact match (highest priority)
2. Semantic similarity using sentence embeddings
3. No resolution (value is preserved)

This component does NOT modify the original schema directly.

This design ensures:
- Traceability
- Auditability
- Safe semantic correction
- Structural integrity preservation

Note
----
This module depends on:
- sentence-transformers
- scikit-learn (cosine_similarity)
"""

from typing import Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import copy


class EntityTypeResolver:
    """
    Resolves entity type filters using deterministic and semantic matching.

    This class normalizes values for filters such as:

        {"attribute": "type", "label": "Place", "value_str": "petshop"}

    into a predefined controlled vocabulary (e.g. "pet_store").

    Parameters
    ----------
    threshold : float, optional (default=0.65)
        Minimum cosine similarity required to accept a semantic match.

    Attributes
    ----------
    valid_types : list[str]
        Controlled vocabulary of accepted entity types.

    model : SentenceTransformer
        Pretrained embedding model used for semantic comparison.

    type_embeddings : np.ndarray
        Precomputed normalized embeddings for valid types.

    threshold : float
        Semantic similarity acceptance threshold.

    Note
    ----
    Embeddings for valid types are computed once during initialization
    to avoid repeated encoding and reduce inference overhead.
    """

    def __init__(self, threshold: float = 0.65):
        self.valid_types = [
            "pet_store",
            "veterinary_care"
        ]

        # Pretrained multilingual model allows flexible user inputs
        self.model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        self.threshold = threshold

        # Precompute normalized embeddings once to avoid
        # repeated encoding during resolution calls (performance optimization)
        self.type_embeddings = self.model.encode(
            self.valid_types,
            normalize_embeddings=True
        )

    def resolve(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve entity type filters inside a schema.

        This method:
        - Preserves the original schema (immutability guarantee)
        - Applies resolution to a working copy
        - Records resolution decisions in an analysis report

        Parameters
        ----------
        schema : dict
            Schema dictionary containing constraints and filters.

        Returns
        -------
        dict
            A dictionary containing:
            - original_schema : dict
            - resolved_schema : dict
            - analysis : list[dict]

        Resolution Strategy
        -------------------
        1. Exact string match (deterministic)
        2. Semantic similarity match (if above threshold)
        3. No correction if similarity is below threshold

        Important
        ---------
        Only the "value_str" field is modified.
        All structural elements remain unchanged.
        """

        # Preserve original schema to guarantee immutability
        original_schema = copy.deepcopy(schema)

        # Work on a separate copy to avoid side effects
        resolved_schema = copy.deepcopy(schema)

        filters = resolved_schema.get("constraints", {}).get("filters", [])

        analysis = []

        for f in filters:

            # Only resolve entity type filters applied to Place nodes
            if f.get("attribute") == "type" and f.get("label") == "Place":

                value = f.get("value_str")
                if value is None:
                    continue

                resolved_value = None
                resolution_strategy = "none"
                resolution_score = 0.0

                # ---------------------------------------------
                # Case 1 — Deterministic exact match (highest confidence)
                # ---------------------------------------------
                if value in self.valid_types:
                    resolved_value = value
                    resolution_score = 1.0
                    resolution_strategy = "exact_match"

                else:
                    # ---------------------------------------------
                    # Case 2 — Semantic Similarity Resolution
                    # ---------------------------------------------

                    # Encode input value
                    value_emb = self.model.encode(
                        [value],
                        normalize_embeddings=True
                    )[0]

                    # Compute cosine similarity against known types
                    sims = cosine_similarity(
                        [value_emb],
                        self.type_embeddings
                    )[0]

                    best_idx = sims.argmax()
                    best_score = float(sims[best_idx])

                    # Accept only if similarity passes threshold
                    if best_score >= self.threshold:
                        resolved_value = self.valid_types[best_idx]
                        resolution_score = best_score
                        resolution_strategy = "semantic_similarity"

                # Apply correction ONLY to the value_str field
                # Structural integrity of the schema is preserved
                if resolved_value:
                    f["value_str"] = resolved_value

                # Record resolution evidence for auditability
                analysis.append({
                    "attribute": "type",
                    "original_value": value,
                    "resolved_value": resolved_value,
                    "resolution_score": resolution_score,
                    "resolution_strategy": resolution_strategy
                })

        return {
            "original_schema": original_schema,
            "resolved_schema": resolved_schema,
            "analysis": analysis
        }
