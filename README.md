# Graph Query Compiler

**From natural language to reliable graph queries — with structured reasoning.**

Graph Query Compiler (GQC) bridges natural language and graph query languages through structured reasoning.

It compiles natural language questions into explicit, executable graph query schemas, enabling reliable and interpretable reasoning over structured data.

GQC reframes query generation as a **structured prediction problem over graph schemas**.

Instead of directly generating answers, the model produces an intermediate semantic representation describing:

- entities
- relationships
- constraints
- aggregations
- multi-hop paths

This representation can be deterministically translated into graph query languages such as Cypher, SQL, Gremlin, or SPARQL.

---

## Example

**Natural language question:**

```text
Which veterinary clinics have rating above 4?
```

**Intermediate representation (simplified):**

```json
{
  "filters": [
    {"attribute": "rating", "operator": ">", "value": 4},
    {"attribute": "type", "operator": "=", "value": "veterinary_care"}
  ],
  "target": "Place",
  "return": ["name"]
}
```

**Generated Cypher:**

```cypher
MATCH (p:Place)
WHERE p.rating > 4
  AND p.type = 'veterinary_care'
RETURN p.name AS name
```

**Pipeline output:**

```json
{
  "status": "success",
  "question": "Which veterinary clinics have rating above 4?",
  "cypher_query": "MATCH (p:Place) WHERE p.rating > 4 AND p.type = 'veterinary_care' RETURN p.name AS name"
}
```

---

## Why this matters

Most LLM-based systems generate queries as plain text, which leads to:

- brittle outputs
- lack of validation
- unpredictable behavior in production

Graph Query Compiler introduces a structured intermediate layer, turning query generation into a deterministic and verifiable process.

This makes it suitable for real-world systems where reliability matters, especially in production environments involving structured data.

---

## Use Cases

- Natural language interfaces for graph databases
- Business intelligence over structured data
- Semantic search over knowledge graphs
- Query generation for Neo4j / SQL systems
- Assistive analytics tools

---

## Key Features

- Structured intermediate representation (IR)
- Deterministic query generation
- Schema-aware validation
- Support for multi-hop graph queries
- Modular pipeline (generation → validation → compilation)

---

## Motivation  

Large Language Models (LLMs) are powerful but often struggle with structured reasoning tasks:

- hallucinated facts
- lack of interpretability
- weak control over constraints
- difficulty performing multi-hop reasoning

Graph Query Compiler addresses these limitations by introducing an explicit intermediate reasoning layer.

This intermediate representation makes it possible to validate, constrain, and execute reasoning steps before they are expressed in natural language.

As a result, the system:

- improves reliability
- enables deterministic execution
- makes reasoning inspectable
- encourages compositional generalization

---

## Core Idea

Graph Query Compiler treats query generation as a program synthesis problem:

Natural language questions are compiled into structured schemas that describe the semantic structure of the query.

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

## Architecture

Graph Query Compiler is organized as a modular pipeline:

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

Each stage is independently controlled, enabling extensibility and debugging.

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

### 3. Data preparation pipeline (Intents → Dataset → Split)

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

---

## Project Structure

High-level overview of the system architecture:

```text
src/
├── compiler/        # Core query compilation pipeline (IR → Cypher)
├── intents/         # Intent generation and semantic validation
├── datasets/        # Dataset generation and preprocessing
├── fine_tuning/     # Model training and inference (QLoRA)
├── config/          # Configuration, schema, and environment
```

---

## Documentation

For a deeper explanation of the design and theoretical motivation:

- [Technical Article](docs/article_graph_query_compiler.md)

---

## Status

Research prototype with a fully working end-to-end pipeline:
natural language → structured schema → executable Cypher query.
