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

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/ricardo-yos/graph-query-compiler
cd graph-query-compiler
```

### 2. Install dependencies

Production install:
```bash
pip install .
```

Development / full environment (recommended):
```bash
pip install -e ".[all]"
```

### 3. Data Preparation Pipeline (Intents → Dataset → Split)

```bash
python -m src.intents.dataset.build_structural_dataset
python -m src.datasets.generation.distilabel_pipeline
python -m src.datasets.splitting.structural_split
```

### 4. Train model (QLoRA fine-tuning)

```bash
python -m src.fine_tuning.training.train_qlora
```

### 5. Run inference (query compiler)

```bash
python -m src.compiler.query_compiler
```

## Recommended Setup

For full reproducibility:

```bash
pip install -e ".[all]"
```

---

## Project Structure

```text
graph-query-compiler/
│
├── src/
│   ├── compiler/                  # Core query compilation pipeline
│   │   ├── codegen/
│   │   │   └── cypher_generator.py
│   │   ├── normalization/
│   │   │   └── normalizer.py
│   │   ├── validation/
│   │   │   └── validator.py
│   │   └── query_compiler.py
│   │
│   ├── config/                   # Configuration files
│   │   ├── datasets/
│   │   │   └── generation.yaml
│   │   ├── fine_tuning/
│   │   │   ├── inference/
│   │   │   │   └── inference_config.yaml
│   │   │   └── training/
│   │   │       └── qlora_config.yaml
│   │   ├── graph/
│   │   │   ├── graph_schema.json
│   │   │   └── schema_loader.py
│   │   ├── intents/
│   │   │   ├── combinatorial.yaml
│   │   │   └── regime_types.yaml
│   │   ├── env_loader.py
│   │   └── paths.py
│   │
│   ├── datasets/                 # Dataset generation and splitting
│   │   ├── generation/
│   │   │   └── distilabel_pipeline.py
│   │   └── splitting/
│   │       └── structural_split.py
│   │
│   ├── fine_tuning/              # Model training and inference
│   │   ├── inference/
│   │   │   └── run_inference.py
│   │   └── training/
│   │       └── train_qlora.py
│   │
│   └── intents/                  # Intent generation and validation
│       ├── dataset/
│       │   └── build_structural_dataset.py
│       ├── dataset_curation/
│       │   └── semantic_bucket_selector.py
│       ├── generation/
│       │   ├── policies/
│       │   │   ├── aggregate_policy.py
│       │   │   ├── filter_policy.py
│       │   │   ├── numeric_policy.py
│       │   │   ├── operator_policy.py
│       │   │   ├── order_policy.py
│       │   │   ├── path_policy.py
│       │   │   ├── return_policy.py
│       │   │   └── value_policy.py
│       │   ├── utils/
│       │   │   ├── attribute_utils.py
│       │   │   └── path_utils.py
│       │   ├── combinatorial_generator.py
│       │   ├── graph_schema_adapter.py
│       │   ├── intent_models.py
│       │   └── structural_config.py
│       ├── reports/
│       │   ├── generate_structural_reports.py
│       │   ├── path_coverage_report.py
│       │   └── save_report.py
│       └── validation/
│           ├── intent_semantic_rules.py
│           └── intent_validator.py
│
├── data/                         # Generated data and artifacts
│   ├── datasets/
│   │   ├── base/
│   │   │   └── questions_base.jsonl
│   │   └── splits/
│   │       ├── train_base.jsonl
│   │       └── val_base.jsonl
│   ├── intents/
│   │   └── structural_intents.jsonl
│   ├── reports/
│   │   └── path_coverage_report.json
│   └── schema/
│       ├── graph_schema.json
│       └── graph_schema_full_reference.json
│
├── docs/                         # Documentation and articles
│   └── article_graph_query_compiler.md
│
├── pyproject.toml
├── README.md
└── LICENSE
```

---

## Documentation

For a detailed explanation of the architecture, design decisions, and theoretical foundations of the project, see:

- [Technical Article](docs/article_graph_query_compiler.md)
