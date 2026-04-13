"""
Graph Query Compiler
====================

End-to-end orchestration of the semantic query compilation pipeline.

This module connects all stages required to transform a natural language
question into an executable Cypher query.

Pipeline Overview
-----------------
    1. LLM Inference        → Generate structured schema from text
    2. Normalization        → Enforce structural consistency
    3. Validation           → Ensure semantic correctness (graph-aware)
    4. Code Generation      → Compile schema into Cypher query

Design Principles
-----------------
- Clear separation of concerns between stages
- Deterministic execution after LLM inference
- Fail-fast validation (invalid queries are rejected early)
- Structured outputs for debugging and observability

Output Contract
---------------
Returns a dictionary with:
- status: "success" or "error"
- intermediate pipeline states (LLM, normalized, validated)
- final Cypher query (if successful)
"""

from sentence_transformers import SentenceTransformer
from src.fine_tuning.inference.run_inference import predict
from src.compiler.normalization.normalizer import SchemaNormalizer
from src.compiler.validation.validator import SchemaValidator, SchemaValidationError
from src.compiler.codegen.cypher_generator import CypherGenerator

import json


class GraphQueryCompiler:
    """
    Orchestrates the full semantic query compilation pipeline.

    This class acts as the main entry point for converting
    natural language into graph queries.
    """

    def __init__(self):
        """
        Initialize shared resources.

        Currently loads a multilingual embedding model,
        which can be reused across future semantic steps
        (e.g., entity resolution, similarity matching).
        """
        self.model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

    def run(self, question: str, debug: bool = True) -> dict:
        """
        Execute the full compilation pipeline.

        Parameters
        ----------
        question : str
            Natural language query from the user
        debug : bool
            Enables debug mode in LLM inference

        Returns
        -------
        dict
            Structured response containing:
            - pipeline outputs (LLM, normalized, validated)
            - final Cypher query (if successful)
            - error information (if failure occurs)
        """

        try:
            # -----------------------------
            # 1 — LLM prediction
            # -----------------------------
            # Converts natural language into a structured schema
            llm_output = predict(question, debug=debug)

            schema_from_llm = llm_output.get("schema")
            if not schema_from_llm:
                raise ValueError("No schema returned from LLM prediction.")

            user_intent = llm_output.get("user_intent")

            # -----------------------------
            # 2 — Schema normalization
            # -----------------------------
            # Enforces structural consistency (types, defaults, format)
            raw_ir = {
                "user_intent": user_intent,
                "schema": schema_from_llm
            }

            normalized_ir = SchemaNormalizer.normalize(raw_ir)

            # -----------------------------
            # 3 — Validation (gatekeeper)
            # -----------------------------
            # Ensures schema is semantically valid against graph structure
            # Critical stage: prevents invalid queries from reaching DB
            SchemaValidator.validate(normalized_ir)

            # Validator is a gatekeeper → does not transform, only validates
            validated_ir = normalized_ir

            # -----------------------------
            # 4 — Cypher generation
            # -----------------------------
            # Deterministically compiles schema into Cypher query
            generator = CypherGenerator(validated_ir)
            cypher_query = generator.generate()

            # -----------------------------
            # Return success
            # -----------------------------
            return {
                "status": "success",
                "question": question,
                "llm_output": llm_output,
                "normalized_schema": normalized_ir,
                "validated_schema": validated_ir,
                "cypher_query": cypher_query
            }

        except SchemaValidationError as e:
            # Known failure: semantic inconsistency
            return {
                "status": "error",
                "stage": "validation",
                "message": str(e),
                "question": question,
                "llm_output": llm_output
            }

        except Exception as e:
            # Unknown failure: infrastructure, inference, etc.
            return {
                "status": "error",
                "stage": "unknown",
                "message": str(e),
                "question": question
            }


if __name__ == "__main__":
    """
    Example execution of the full pipeline.

    Demonstrates:
    - End-to-end compilation
    - JSON output structure
    - Pretty-printed Cypher query
    """
    pipeline = GraphQueryCompiler()

    question = "Quais clinicas veterinárias possuem nota acima de 4?"

    output = pipeline.run(question)

    # Full structured output (debugging / logging)
    print(json.dumps(output, indent=2, ensure_ascii=False))

    # Pretty print Cypher query
    if output.get("status") == "success":
        print("\n" + "="*50)
        print("Cypher Query:\n")
        print(output["cypher_query"])
        print("="*50)
