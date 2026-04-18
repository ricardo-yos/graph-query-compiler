# Graph Query Compiler: Structured Reasoning for Reliable Natural Language Interfaces to Knowledge Graphs

## 1. Introduction

Large language models (LLMs) have demonstrated strong capabilities in generating fluent and contextually coherent text. However, their ability to perform structured reasoning over constrained domains remains limited. Tasks that require consistency with an underlying schema—such as graph query generation—often expose these limitations, as models may produce outputs that are syntactically plausible but semantically invalid. Even when LLMs appear to perform reasoning, this process remains largely implicit and does not reliably enforce domain-specific constraints.

These limitations become critical in scenarios where multiple attributes, relationships, and filtering conditions must be composed while preserving structural validity. In such cases, LLMs frequently generate outputs that violate schema constraints, introduce incompatible attribute combinations, or misalign the intended query structure with its natural language representation.

Approaches that rely solely on language modeling cannot reliably enforce structural constraints. As a result, they often produce inconsistent or semantically invalid outputs, limiting their effectiveness in tasks such as semantic parsing, dataset generation, and natural language interfaces over structured data systems.

To address this gap, we propose the Graph Query Compiler (GQC), a deterministic pipeline that separates structural reasoning from linguistic realization. The system first generates candidate query intents through controlled combinatorial processes and enforces schema consistency via a semantic validation layer, before mapping these validated structures into natural language. By enforcing structural correctness prior to language generation, GQC mitigates common failure modes of LLMs and enables the creation of high-quality, semantically consistent datasets.

This approach provides a more reliable foundation for tasks that depend on structured reasoning, while preserving the expressive flexibility of natural language.

## 2. System Overview

The Graph Query Compiler (GQC) is a deterministic pipeline that generates and validates structured query representations before compiling them into executable graph queries. By explicitly separating structural reasoning from linguistic realization, the system ensures that all outputs are consistent with a predefined schema and can be reliably executed in graph database environments.

### Stage 1 — Dataset Generation
In the first stage, the system generates structured query intents through controlled combinatorial processes within a predefined and schema-constrained domain. These intents are then used to construct a dataset that aligns schema-consistent representations with their corresponding natural language expressions.

**Subcomponents**

* **Intent Generation**: Structured query intents are generated based on a predefined schema. These intents represent composable query specifications, including entities, attributes, relationships, and filtering conditions. The generation process explores the space of valid query structures while maintaining control over combinatorial complexity.

* **Dataset Construction**: Generated intents are transformed into training examples by pairing structured representations with their natural language counterparts. This ensures alignment between structure and language, enabling the creation of high-quality supervised datasets.

### Stage 2 — Model Adaptation and Inference
In the second stage, the generated dataset is used to adapt language models via fine-tuning. The model learns to map natural language inputs to structured query representations, enabling accurate schema inference from user queries.

**Subcomponents**

* **Fine-tuning**: The model is trained on the generated dataset to learn the correspondence between natural language and structured query intents.

* **Schema Inference**: At inference time, the model predicts structured representations from natural language questions, effectively recovering the underlying query intent.

### Stage 3 — Query Compilation
In the final stage, the inferred structured representation is transformed into an executable query, such as Cypher. This ensures that outputs are not only linguistically meaningful but also operationally valid.

**Subcomponents**

* **Query Translation (Cypher Generation)**: Structured query representations are compiled into executable queries, ensuring compatibility with graph database systems.

At a high level, the pipeline can be summarized as:

```text
Intent Generation → Dataset Construction → Model Adaptation → Query Compilation
```

This modular architecture allows each stage to operate independently while contributing to a unified objective: enabling reliable, interpretable, and structurally consistent query generation with direct executability in graph database systems.

This design adopts a structure-first approach, ensuring correctness before natural language generation.

## 3. Methodology

This section describes the core components of the Graph Query Compiler (GQC), focusing on how structured query representations are generated, validated, and transformed into executable queries. The methodology is designed to enforce schema consistency and control the combinatorial space of possible queries, ensuring both diversity and structural correctness.

### 3.1 Intent Representation

In GQC, queries are represented as structured intents that explicitly encode the components of a graph query. An intent is a schema-aware representation composed of entities, attributes, relationships, and filtering conditions.

This intermediate representation serves as a bridge between natural language and executable queries. By decoupling query structure from both linguistic expression and execution syntax, the system enables controlled manipulation, validation, and transformation of query logic.

An example of a structured query intent is shown below:

```json
{
  "intent": {
    "type": "retrieve",
    "modifiers": ["filter"],
    "regime": "lookup_basic"
  },
  "schema_spec": {
    "target": {
      "label": "Place"
    },
    "path": [],
    "filters": [
      {
        "node_label": "Place",
        "attribute": "rating",
        "operator": ">",
        "value": 4
      },
      {
        "node_label": "Place",
        "attribute": "type",
        "operator": "=",
        "value": "veterinary_care"
      }
    ],
    "order_by": null,
    "limit": null,
    "aggregate": null,
    "return_attributes": ["name", "rating"]
  }
}
```

This representation separates high-level intent semantics from schema-specific query structure. The `intent` field encodes the type of operation and its modifiers, while `schema_spec` defines the structural components of the query, including the target entity, filtering conditions, and returned attributes. This separation enables precise and reliable manipulation of query generation, validation, and transformation, ensuring that all queries remain consistent with the underlying schema.

### 3.2 Intent Generation

Intent generation in GQC is performed through a schema-driven combinatorial pipeline over a predefined graph schema, where nodes, attributes, and relationships define the space of valid query structures.

The objective is to systematically construct structured query representations (`intent` + `schema_spec`) through controlled combinations of traversal paths and structural components, while ensuring both diversity and structural correctness.

The generation process involves exploring graph traversal paths with varying depths and composing structural operators such as filtering, aggregation, and ordering.

Given this schema, candidate intents are generated by composing:

* target nodes
* attribute-based filters
* relationship paths
* structural operators such as ordering, limits, and aggregation

To structure the generation process, intent generation is performed with respect to predefined structural regimes, where each regime defines a class of query patterns with a specific level of structural complexity. These regimes capture patterns such as simple lookups, multi-attribute filtering, relational queries, and aggregation-based queries.

By generating intents within these regimes, the system gains explicit control over the complexity of the resulting queries, ensuring balanced coverage across different levels of structural difficulty and preventing the dataset from being biased toward simpler or more frequent query patterns.

Within each regime, the generation process is further guided by field-level policies defined over both the intent and schema specifications. These policies regulate how individual components—such as attributes, operators, and values—can be combined, ensuring semantic coherence and eliminating invalid or unrealistic query patterns. By enforcing these localized constraints, the system achieves finer control over the generation space while maintaining flexibility.

This design introduces multiple levels of control over the generation process, ranging from high-level structural regimes to fine-grained constraints on individual query components.

To ensure validity, generated candidates are subject to structural and semantic constraints, including schema compatibility, type consistency, and limits on query complexity.

To promote diversity, a semantic balancing strategy is applied to prevent overrepresentation of specific query patterns, ensuring broader coverage of the query space.

Unlike probabilistic approaches, this method explicitly constructs query structures within a controlled and deterministic search space. As a result, all generated intents are schema-consistent by design and provide high structural coverage for downstream dataset construction and model training.
