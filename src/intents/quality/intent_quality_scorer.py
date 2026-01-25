"""
Intent Quality Scorer
=====================

Deterministic quality scoring for validated graph query intents.

This module assigns a normalized quality score to each intent based on
its structural richness, semantic usefulness, and suitability for:

- Natural Language question generation
- Cypher query synthesis
- Dataset curation and filtering

Important
---------
- This scorer assumes intents are already validated.
- No structural or semantic validation is performed here.
- Scores are deterministic and fully explainable.
"""

from typing import Dict, Tuple, List


# --------------------------------------------------
# Intent quality scorer
# --------------------------------------------------

class IntentQualityScorer:
    """
    Assign a quality score to a validated intent.

    The score reflects how expressive, useful, and non-trivial
    an intent is for downstream tasks such as NL generation
    and Cypher synthesis.
    """

    # --------------------------------------------------
    # Base priors aligned with generated dataset
    # --------------------------------------------------

    INTENT_BASE_SCORE = {
        "rank": 0.90,
        "count": 0.85,
        "retrieve": 0.75,
        "filter": 0.70,
    }

    # --------------------------------------------------
    # Main scoring entry point
    # --------------------------------------------------

    def score(self, intent: Dict) -> Tuple[float, List[str]]:
        """
        Compute a quality score for a validated intent.

        Parameters
        ----------
        intent : dict
            A validated intent specification.

        Returns
        -------
        Tuple[float, List[str]]
            - Final quality score (clamped to [0, 1])
            - List of human-readable scoring reasons
        """
        score = 0.0
        reasons: List[str] = []

        intent_type = intent.get("user_intent", "unknown")

        # --------------------------------------------------
        # 1. Base intent prior
        # --------------------------------------------------

        base = self.INTENT_BASE_SCORE.get(intent_type, 0.30)
        score += base
        reasons.append(f"base_score({intent_type})={base}")

        # --------------------------------------------------
        # 2. Structural signals
        # --------------------------------------------------

        constraints = intent.get("constraints") or {}

        filters = constraints.get("filters") or []
        order_by = constraints.get("order_by") or []
        limit = constraints.get("limit") or []

        path = intent.get("path") or []
        known = intent.get("known") or []

        # ---- Filters
        if filters:
            score += 0.15
            reasons.append("has_filters(+0.15)")

            if len(filters) > 1:
                score += 0.05
                reasons.append("multi_filters(+0.05)")

        # ---- Known inputs
        if known:
            score += 0.10
            reasons.append("has_known(+0.10)")

        # ---- Path richness
        if path:
            score += 0.20
            reasons.append("has_path(+0.20)")

            if len(path) > 1:
                score += 0.10
                reasons.append("multi_hop_path(+0.10)")

        # ---- Ordering
        if order_by:
            score += 0.10
            reasons.append("has_order_by(+0.10)")

        # ---- Limit
        if isinstance(limit, list) and limit:
            score += 0.05
            reasons.append("has_limit(+0.05)")

        # --------------------------------------------------
        # 3. Intent-specific penalties
        # --------------------------------------------------

        if intent_type == "retrieve":
            if not path and not filters and not known:
                score -= 0.25
                reasons.append("trivial_retrieve(-0.25)")

            if known and not path and not filters:
                score -= 0.25
                reasons.append("lookup_like_retrieve(-0.25)")

        if intent_type == "filter" and not filters:
            score -= 0.20
            reasons.append("filter_without_filters(-0.20)")

        if intent_type == "filter" and path and not filters:
            score -= 0.30
            reasons.append("filter_with_path_but_no_constraints(-0.30)")

        # --------------------------------------------------
        # 4. Redundancy & semantic penalties
        # --------------------------------------------------

        # Repeated relationships in path
        for i in range(len(path) - 1):
            if path[i].get("rel") == path[i + 1].get("rel"):
                score -= 0.10
                reasons.append("redundant_path_relation(-0.10)")
                break

        # Tautological range filters
        if len(filters) == 2:
            f1, f2 = filters
            if (
                f1.get("label") == f2.get("label")
                and f1.get("attribute") == f2.get("attribute")
                and {f1.get("operator"), f2.get("operator")}
                <= {">", ">=", "<", "<="}
            ):
                score -= 0.10
                reasons.append("tautological_range_filter(-0.10)")

        # --------------------------------------------------
        # 5. Expressiveness bonuses
        # --------------------------------------------------

        if intent_type == "rank" and path:
            score += 0.05
            reasons.append("rank_over_path(+0.05)")

        if intent_type == "filter" and path:
            score += 0.05
            reasons.append("filter_over_path(+0.05)")

        # --------------------------------------------------
        # Clamp and return
        # --------------------------------------------------

        score = max(0.0, min(1.0, score))
        return round(score, 3), reasons
