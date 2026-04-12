# Graph Query Compiler

**Graph Query Compiler (GQC)** is a structured reasoning interface between natural language and knowledge graphs.

It compiles natural language questions into explicit, executable graph query schemas, enabling reliable and interpretable reasoning over structured data.

Instead of directly generating answers from text, the model learns to produce an intermediate semantic representation describing:

- entities
- relationships
- constraints
- aggregations
- multi-hop paths

This representation can be deterministically translated into graph query languages such as Cypher, SQL, Gremlin, or SPARQL.

---

## Motivation  

Large Language Models (LLMs) are powerful but often face limitations when applied to structured reasoning tasks:

- hallucinated facts
- lack of interpretability
- weak control over structural constraints
- difficulty performing multi-hop reasoning
- limited reliability in compositional queries

Graph Query Compiler addresses these limitations by introducing an explicit intermediate reasoning layer between natural language and executable graph queries.

Instead of optimizing the model to directly produce answers, GQC optimizes the model to produce structured reasoning steps that can be validated and executed.

By forcing the model to produce structured representations, GQC:

- improves reliability of generated queries
- enables deterministic execution on knowledge graphs
- makes reasoning interpretable and inspectable
- encourages compositional generalization
- reduces dependence on memorization of specific questions

---

## Core Idea

Graph Query Compiler treats query generation as a program synthesis problem:

Natural language questions are compiled into structured intent schemas that describe the semantic structure of the query.

These schemas can then be translated into executable graph queries.

Pipeline overview:

```text
Natural Language Question
        ↓
Structured Intent Schema
        ↓
Graph Query (Cypher / SQL / etc.)
        ↓
Execution on Knowledge Graph
        ↓
Grounded Answer
```

---

## Example

The following example illustrates the full pipeline from natural language to executable query:

Natural language question:

```text
Which veterinary clinics have rating above 4?
```

Generated intent schema:

```json
{
  "user_intent": "retrieve",
  "schema": {
    "aggregate": null,
    "filters": [
      {
        "attribute": "rating",
        "node_label": "Place",
        "operator": ">",
        "value_float": null,
        "value_int": 4,
        "value_str": null
      },
      {
        "attribute": "type",
        "node_label": "Place",
        "operator": "=",
        "value_float": null,
        "value_int": null,
        "value_str": "veterinary_care"
      }
    ],
    "limit": null,
    "order_by": null,
    "path": [],
    "return_attributes": [
      "name"
    ],
    "target": {
      "label": "Place"
    }
  }
}
```

Compiled graph query (Cypher):

```cypher
MATCH (p:Place)
WHERE p.rating > 4
  AND p.type = "veterinary_care"
RETURN p.name AS name
```

---

## Architecture

Graph Query Compiler is organized as a modular pipeline separating structural reasoning from language generation.

High-level flow:

```text
Graph Schema Definition
        ↓
Structural Intent Generation
        ↓
Semantic Validation
        ↓
Natural Language Question Generation
        ↓
Training Dataset (question → intent schema)
        ↓
QLoRA Fine-tuning
        ↓
Inference
        ↓
Schema Compilation
        ↓
Graph Query Execution
```

This pipeline separates structure generation, validation, and execution, ensuring that each stage can be independently controlled and improved.

---

## Key Components

### Intent Schema
Structured representation describing the semantic structure of a query, including entities, filters, constraints, and expected outputs.

### Structural Generator
Generates valid combinations of entities, relationships, and constraints based on the underlying graph schema.

### Semantic Validator
Applies rules to ensure generated intents are logically consistent and compatible with the graph schema.

### Question Generator
Transforms structured intents into natural language questions.

### Training Pipeline
Fine-tunes the model to map questions to structured schemas.

### Query Compiler
Converts validated intent schemas into executable graph queries (e.g., Cypher, SQL, Gremlin).

---
