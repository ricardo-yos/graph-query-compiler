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

The generation process involves exploring graph traversal paths with varying depths and composing operators such as filtering, aggregation, and ordering.

Given this schema, candidate intents are generated by composing:

* target nodes
* attribute-based filters
* relationship paths
* operators such as ordering, limits, and aggregation

To structure the generation process, intent generation is performed with respect to predefined structural regimes, where each regime defines a class of query patterns with a specific level of query complexity. These regimes capture patterns such as simple lookups, multi-attribute filtering, relational queries, and aggregation-based queries.

Each regime implicitly defines which components are present in the query representation, such as the inclusion of filters, relational paths, or aggregation operators. In this way, regimes not only categorize query patterns but also constrain the composition of generated intents.

By generating intents within these regimes, the system gains explicit control over the complexity of the resulting queries, ensuring balanced coverage across different levels of difficulty and preventing the dataset from being biased toward simpler or more frequent query patterns.

Within each regime, the generation process is guided by field-level policies defined over both the intent and schema specifications. These policies regulate how individual components—such as attributes, operators, and values—can be combined, as well as which values and operators are considered valid within a given context, ensuring semantic coherence and eliminating invalid or unrealistic query patterns.

Operating at the level of individual schema components, these policies enable fine-grained control over how attributes, operators, and values are assigned within each part of the query representation. By constraining the admissible combinations at this level, the system effectively limits the combinatorial explosion inherent to the generation process while preserving diversity.

This design introduces multiple levels of control over the generation process, ranging from high-level structural regimes to fine-grained constraints on individual query components.

To ensure validity, generated candidates are subject to structural and semantic constraints, including schema compatibility, type consistency, and limits on query complexity.

To promote diversity, a semantic balancing strategy is applied to prevent overrepresentation of specific query patterns, ensuring more comprehensive coverage of the query space.

Unlike probabilistic approaches, this method explicitly constructs query structures within a controlled and deterministic search space. As a result, all generated intents are schema-consistent by design and provide high structural coverage for downstream dataset construction and model training.

### 3.3 Semantic Validation

While the intent generation process enforces structural constraints during construction, a dedicated validation layer is responsible for filtering generated intents before they are accepted into downstream stages of the pipeline.

This validation layer operates over the structured intent representation (`intent` + `schema_spec`) and ensures that each candidate is structurally consistent, semantically meaningful, compatible with natural language expression, and aligned with domain constraints.

Validation is performed declaratively using a centralized set of semantic rules, which define the domain-level conditions under which entities, attributes, operators, and values are considered valid and meaningful.

These rules capture constraints such as:

* valid target structures and return attributes
* compatibility between attributes and operators
* semantic consistency of filter definitions
* validity of ordering and aggregation clauses
* coherence of traversal paths
* alignment with attribute semantic types

Importantly, this design separates semantic knowledge from execution logic. The rule set defines *what is valid*, while the validator is responsible for enforcing these constraints by filtering out invalid or incoherent query representations.

The validator operates strictly as a verification layer. It does not generate, modify, or rank intents, nor does it perform schema traversal. Instead, it acts as a gatekeeping mechanism that ensures only valid intents are propagated through the pipeline.

This separation improves modularity and maintainability, allowing domain knowledge to evolve independently from validation logic.

By combining schema-level consistency checks with rule-based semantic validation, the system ensures that all accepted intents are coherent, interpretable, and suitable for downstream tasks such as dataset construction and model training.

Unlike purely generative approaches, where invalid outputs must be corrected or discarded post hoc, this method integrates validation as a first-class filtering stage. As a result, correctness is not assumed but systematically enforced before any downstream use.

### 3.4 Dataset Construction

Once validated intents are obtained, the system constructs a supervised dataset by pairing structured query representations with their corresponding natural language expressions.

Each data sample consists of a tuple:

- natural language query
- structured intent representation (`intent` + `schema_spec`)

This pairing establishes a direct mapping between linguistic input and structured query semantics, enabling the training of models for natural language to query translation.

Natural language expressions are generated based on the underlying intent representation, ensuring that each query accurately reflects the structure and semantics encoded in the corresponding schema specification. This preserves alignment between form (language) and meaning (query structure), which is critical for downstream learning tasks.

The quality of the resulting dataset is directly influenced by the generation and validation processes. Since all intents are constructed within a schema-constrained space, guided by structural regimes and field-level policies, and filtered through a semantic validation layer, the dataset inherits strong guarantees of consistency, coherence, and validity.

In addition, the semantic balancing strategy applied during intent generation ensures that the dataset maintains a diverse distribution of query patterns. This prevents overrepresentation of specific structures and promotes broader coverage across different query types and complexity levels.

As a result, the constructed dataset provides:

* high structural diversity across query patterns
* balanced representation of complexity levels
* strong alignment between natural language and structured queries
* elimination of invalid or incoherent samples

Unlike datasets derived from raw text or weak supervision, this approach produces high-quality training data with explicit structural grounding and minimal noise.

This dataset serves as the foundation for model adaptation, enabling language models to learn accurate mappings from natural language inputs to structured query representations.

### 3.5 Model Adaptation

The dataset constructed from validated intents is used to adapt language models to the task of mapping natural language queries to structured query representations.

This adaptation is performed through supervised fine-tuning, where the model learns to predict structured intents (`intent` + `schema_spec`) given a natural language input. The objective is to enable accurate schema inference from user queries, bridging the gap between unstructured language and executable query logic.

During training, each sample provides a direct alignment between a natural language expression and its corresponding structured representation. This explicit supervision allows the model to learn compositional mappings between linguistic patterns and query components, such as entities, attributes, filters, and operators.

Unlike standard language modeling, which relies on implicit pattern learning, this approach introduces a structured prediction task where correctness can be explicitly evaluated at the level of the generated intent.

At inference time, the fine-tuned model receives a natural language query and predicts the corresponding structured representation. This predicted intent serves as an intermediate, interpretable form that can be validated and compiled into an executable query.

The use of structured representations provides several advantages:

* **Interpretability**: The predicted intent can be inspected and analyzed before execution.
* **Controllability**: Errors can be identified and corrected at the structural level.
* **Modularity**: The system decouples language understanding from query execution.
* **Robustness**: The model operates within a schema-constrained space, reducing invalid outputs.

Furthermore, the high-quality dataset produced by the GQC pipeline—characterized by structural consistency, semantic validity, and balanced coverage—enables more reliable learning compared to datasets derived from noisy or weakly supervised sources.

As a result, the adapted model is better equipped to perform accurate and consistent natural language to query translation, forming a critical component of the overall system.
