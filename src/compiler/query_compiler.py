"""
Graph Query Compiler
====================

Orchestrates the full semantic query compilation pipeline for graph-based queries.

This pipeline handles:

1. LLM prediction of an intermediate schema from a natural language question.
2. Normalization of the schema to ensure required fields and formats.
3. Semantic resolution, validating and refining attributes, types, and operators.
4. Cypher query generation for execution against a graph database.

Input
-----
question : str
    Natural language user query.

debug : bool, optional
    If True, prints debug information during LLM prediction (default: True).

Output
------
dict
    Dictionary containing:
        - question           : Original question
        - llm_output         : Raw schema and intent from LLM
        - normalized_schema  : Schema after structural normalization
        - semantic_output    : Schema after semantic resolution
        - cypher_query       : Generated Cypher query string
"""

from sentence_transformers import SentenceTransformer
from src.fine_tuning.inference.run_inference import predict
from src.compiler.normalization.normalizer import SchemaNormalizer
from src.compiler.semantic.semantic_pipeline import SemanticResolutionPipeline
from src.compiler.semantic.resolvers.entity_resolver import EntityResolver
from src.compiler.semantic.resolvers.entity_type_resolver import EntityTypeResolver
from src.compiler.semantic.resolvers.operator_semantic_resolver import OperatorSemanticResolver
from src.compiler.codegen.cypher_generator import CypherGenerator
from copy import deepcopy
import json


class GraphQueryCompiler:
    """
    Full orchestration of the graph query compilation pipeline.
    """

    def __init__(self):
        """
        Initialize embedding model, semantic resolvers, and the full pipeline.
        """
        # Load shared multilingual embedding model
        self.model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        # Initialize semantic resolvers
        self.entity_resolver = EntityResolver(model=self.model)
        self.entity_type_resolver = EntityTypeResolver(model=self.model)
        self.operator_resolver = OperatorSemanticResolver(model=self.model)

        # Compose full semantic pipeline
        self.semantic_pipeline = SemanticResolutionPipeline(
            entity_resolver=self.entity_resolver,
            entity_type_resolver=self.entity_type_resolver,
            operator_resolver=self.operator_resolver
        )

    def run(self, question: str, debug: bool = True) -> dict:
        """
        Execute the complete graph query compilation pipeline for a single question.

        Parameters
        ----------
        question : str
            Natural language question to compile into a graph query.
        debug : bool
            Enable debug outputs for LLM inference (default: True).

        Returns
        -------
        dict
            Complete pipeline outputs including LLM schema, normalized schema,
            semantic resolution, and generated Cypher query.
        """

        # -----------------------------
        # 1 — LLM prediction
        # -----------------------------
        llm_output = predict(question, debug=debug)
        schema_from_llm = llm_output.get("schema")
        if not schema_from_llm:
            raise ValueError("No schema returned from LLM prediction.")

        user_intent = llm_output.get("user_intent")

        # -----------------------------
        # 2 — Schema normalization
        # -----------------------------
        normalized_schema = {
            "user_intent": user_intent,
            "schema": schema_from_llm
        }
        normalized_schema_content = SchemaNormalizer.normalize(normalized_schema)

        # -----------------------------
        # 3 — Semantic resolution
        # -----------------------------
        semantic_output = self.semantic_pipeline.resolve(
            question=question,
            llm_schema=normalized_schema_content["schema"]
        )

        # -----------------------------
        # 4 — Reconstruct IR for Cypher generation
        # -----------------------------
        ir_for_cypher = {
            "user_intent": user_intent,
            "schema": semantic_output.get("resolved_schema"),
        }

        # -----------------------------
        # 5 — Cypher generation
        # -----------------------------
        generator = CypherGenerator(ir_for_cypher)
        cypher_query = generator.generate()

        # -----------------------------
        # Return all outputs
        # -----------------------------
        return {
            "question": question,
            "llm_output": llm_output,
            "normalized_schema": normalized_schema,
            "semantic_output": semantic_output,
            "cypher_query": cypher_query
        }


if __name__ == "__main__":
    pipeline = GraphQueryCompiler()
    question = "Quais os petshops do bairro Jardim que possuem nota acima de 4?"
    output = pipeline.run(question)
    print(json.dumps(output, indent=2, ensure_ascii=False))
