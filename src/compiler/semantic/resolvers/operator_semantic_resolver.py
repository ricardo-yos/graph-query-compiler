"""
Operator Semantic Resolver
==========================

Performs semantic validation and correction of relational operators
(>, <, >=, <=, =) applied to numeric attributes in a compiled schema.

This component operates as a semantic enrichment stage in the
compilation pipeline. It does not generate filters, but validates
and corrects operator semantics inferred earlier in the pipeline.

Resolution Strategy
-------------------
1. Detect the semantic anchor related to the numeric attribute
   inside the user question using a local semantic retriever.
2. Extract a contextual window around the anchor.
3. Compute cosine similarity between the window embedding and
   precomputed mean operator vectors.
4. Apply correction only if confidence exceeds a configurable threshold.

Component Role
--------------
- Enrichment stage (non-destructive)
- Operator semantic validator
- Confidence-based correction mechanism

Input
-----
question : str
    Natural language user query.

schema_state : dict
    Schema representation containing numeric filters that may require
    operator validation or correction.

Output
------
dict
    Updated schema representation where numeric operators are
    semantically validated and corrected when necessary.
    Includes resolution metadata for analysis and observability.

Design Guarantees
-----------------
- Only numeric attributes are evaluated.
- Operator correction is applied only if confidence >= threshold.
- The original_schema reference is preserved.
- All modifications are applied to a deep copy.
- All resolution decisions are logged for observability.

Dependencies
------------
- numpy
- scikit-learn (cosine_similarity)
- SemanticLocalRetriever (semantic anchor detection)
"""

from typing import Dict, Any, Optional
import copy
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.compiler.semantic.services.anchor_window_service import SemanticLocalRetriever


class OperatorSemanticResolver:
    """
    Validates and corrects relational operators applied to numeric
    attributes using local semantic context and operator embeddings.

    Parameters
    ----------
    model : SentenceTransformer, optional (default=None)
        Embedding model compatible with `.encode()`.
        If None, loads a multilingual MiniLM model automatically.

    window_size : int, default=5
        Number of tokens extracted around the semantic anchor.

    threshold : float, default=0.55
        Minimum cosine similarity required to accept correction.

    Notes
    -----
    Operator representations are computed as mean embeddings of
    predefined prototype expressions.
    """

    NUMERIC_ATTRIBUTES = {"rating", "num_reviews"}

    ATTRIBUTE_DESCRIPTIONS = {
        "rating": "nota ou avaliação dada ao estabelecimento",
        "num_reviews": "quantidade ou número total de avaliações recebidas"
    }

    OPERATOR_PROTOTYPES = {
        ">": ["maior que", "acima de", "superior a"],
        ">=": ["maior ou igual a", "pelo menos", "no mínimo", "igual ou acima de"],
        "<": ["menor que", "abaixo de", "inferior a"],
        "<=": ["menor ou igual a", "até", "no máximo", "igual ou abaixo de"],
        "=": ["igual a", "exatamente", "valor igual a"]
    }

    def __init__(
        self,
        model: Optional[SentenceTransformer] = None,
        window_size: int = 5,
        threshold: float = 0.55
    ):
        # Use injected model if provided; otherwise load default multilingual model
        self.model = model or SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        self.threshold = threshold

        # Semantic anchor detector used to extract local context
        self.retriever = SemanticLocalRetriever(
            model=self.model,
            window_size=window_size
        )

        # Precompute mean operator vectors for stable semantic comparison
        self.operator_vectors = self._build_operator_vectors()

    def _build_operator_vectors(self) -> Dict[str, np.ndarray]:
        """
        Build normalized mean embedding vectors for each operator prototype.

        Returns
        -------
        dict
            Mapping of operator symbol -> normalized mean embedding vector.
        """
        operator_vectors = {}

        for op, expressions in self.OPERATOR_PROTOTYPES.items():

            embs = self.model.encode(
                expressions,
                normalize_embeddings=True
            )

            # Mean vector represents semantic centroid of operator expressions
            mean_vector = np.mean(embs, axis=0)

            # Safe normalization to maintain cosine consistency
            norm = np.linalg.norm(mean_vector)
            if norm != 0:
                mean_vector = mean_vector / norm

            operator_vectors[op] = mean_vector

        return operator_vectors

    def _clean_window(self, text: str) -> str:
        """
        Remove numeric values from window text before operator inference.

        This avoids numeric tokens dominating the embedding signal.
        """
        text = re.sub(r"\d+([.,]\d+)?", "", text)
        return text.strip().lower()

    def _resolve_operator(self, window_text: str):
        """
        Infer the most likely relational operator from semantic window text.

        Returns
        -------
        tuple
            (best_operator, confidence_score)
        """

        if not window_text:
            return None, 0.0

        clean_window = self._clean_window(window_text)

        if not clean_window:
            return None, 0.0

        window_emb = self.model.encode(
            [clean_window],
            normalize_embeddings=True
        )[0]

        scores = {}

        # Compare window embedding against each operator centroid
        for op, op_vector in self.operator_vectors.items():
            score = float(
                cosine_similarity(
                    [window_emb],
                    [op_vector]
                )[0][0]
            )
            scores[op] = score

        best_operator = max(scores, key=scores.get)
        best_score = scores[best_operator]

        return best_operator, best_score

    def resolve(
        self,
        question: str,
        schema_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate and correct relational operators in a compiled schema.

        Parameters
        ----------
        question : str
            Original natural language query.
        schema_state : dict
            Output from previous pipeline stage containing schema and analysis.

        Returns
        -------
        dict
            Updated pipeline result including operator validation analysis.
        """

        # Preserve original schema safely (defensive copy)
        original_schema = copy.deepcopy(schema_state["original_schema"])

        # Work on a safe copy to avoid side effects
        resolved_schema = copy.deepcopy(schema_state["resolved_schema"])

        operator_analysis = []

        filters = resolved_schema.get("constraints", {}).get("filters", [])

        for f in filters:

            attribute = f.get("attribute")

            # Only process numeric attributes
            if attribute not in self.NUMERIC_ATTRIBUTES:
                continue

            description = self.ATTRIBUTE_DESCRIPTIONS.get(attribute)
            if not description:
                continue

            # Step 1 — Detect semantic anchor for attribute
            anchor_result = self.retriever.detect_anchor(
                question,
                description
            )

            window_text = anchor_result.get("window_text")

            if not window_text:
                continue

            # Step 2 — Infer operator from local semantic window
            detected_operator, confidence = self._resolve_operator(window_text)

            original_operator = f.get("operator")

            validated = confidence >= self.threshold
            corrected = False

            # Apply correction only if validated and different
            if validated and detected_operator and detected_operator != original_operator:
                f["operator"] = detected_operator
                corrected = True

            # Always log operator resolution for observability
            operator_analysis.append({
                "attribute": attribute,
                "original_operator": original_operator,
                "resolved_operator": detected_operator,
                "validated": validated,
                "corrected": corrected,
                "anchor_text": anchor_result.get("anchor_text"),
                "anchor_similarity": anchor_result.get("similarity"),
                "window_text": window_text,
                "operator_confidence": round(confidence, 3),
                "strategy": "semantic_mean_operator_vector"
            })

        return {
            "original_schema": original_schema,
            "resolved_schema": resolved_schema,
            "analysis": operator_analysis
        }
