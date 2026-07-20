# Graph Query Compiler: Structured Reasoning for Natural Language Interfaces to Knowledge Graphs

## Abstract

Generating reliable graph queries from natural language requires preserving both linguistic meaning and strict structural constraints. Large language models can produce fluent outputs, but they may generate structurally invalid or semantically inconsistent queries when operating without explicit control mechanisms.

This document presents the Graph Query Compiler (GQC) v1, an experimental framework for controlled natural language to graph query translation. The system separates structural query reasoning from linguistic realization by combining schema-driven intent generation, controlled dataset construction, model adaptation, semantic validation, and deterministic query compilation.

A central aspect of GQC v1 is a structure-first pipeline, where the space of valid query structures is explicitly modeled before language generation. Structural regimes and field-level policies guide the generation process, while validation mechanisms enforce consistency between entities, attributes, operators, relationships, and query semantics.

The current version investigates the feasibility of schema-aware natural language interfaces for graph databases. Experimental analysis indicates that the approach supports more consistent structured representations, while also revealing remaining challenges related to linguistic generalization, relational reasoning, and semantic ambiguity.

## 1. Introduction

### 1.1 Motivation

Natural language interfaces have the potential to simplify access to structured data systems by allowing users to interact with databases using everyday language instead of specialized query languages.

However, when applied to graph databases, this task becomes significantly more challenging. Graph queries require preserving not only the meaning expressed in natural language, but also the structural constraints defined by the underlying schema, including entities, relationships, attributes, operators, and query composition rules.

Recent advances in large language models (LLMs) have demonstrated strong capabilities in natural language understanding and generation. Nevertheless, directly generating executable queries with LLMs remains unreliable, as models may produce outputs that are syntactically valid but structurally inconsistent, semantically incorrect, or incompatible with the database schema.

This motivates the exploration of approaches that combine the flexibility of language models with explicit structural control mechanisms.

### 1.2 Problem Definition

The main problem addressed by the Graph Query Compiler (GQC) is the reliable translation of natural language questions into structured graph query representations.

A direct natural language-to-query approach must solve multiple challenges simultaneously:

- identifying the intended entities and relationships;
- mapping linguistic expressions to schema attributes;
- preserving operator semantics;
- composing filters, aggregations, rankings, and traversals correctly;
- generating outputs compatible with execution constraints.

These challenges become increasingly difficult as query complexity grows. Small semantic differences in natural language can result in fundamentally different query structures. For example, expressions involving comparisons, superlatives, or aggregation may require different structural representations even when they appear linguistically similar.

Therefore, the problem is not only generating a query that appears plausible, but generating a schema-consistent representation whose semantics can be validated before execution.

### 1.3 Objectives

The objective of the Graph Query Compiler (GQC) is to develop and evaluate a controlled framework for natural language to graph query translation based on an intermediate structured representation.

The main objectives of GQC v1 are:

1. **Define a schema-aware intermediate representation**
   
   Create a structured query format capable of representing targets, relationships, filters, aggregations, ordering, limits, and returned attributes.

2. **Enable controlled dataset generation**
   
   Develop a schema-driven generation pipeline capable of producing diverse training examples while maintaining structural and semantic consistency.

3. **Introduce explicit validation mechanisms**
   
   Apply semantic and structural validation before model training and query compilation to reduce invalid representations.

4. **Evaluate model-based intent prediction**
   
   Analyze the ability of adapted language models to map natural language questions into structured graph query intents.

5. **Separate language understanding from query execution**
   
   Use deterministic compilation from validated structured intents into executable graph queries, improving reliability and interpretability.

GQC v1 focuses on establishing the foundations of a controlled pipeline for schema-aware graph query generation, while identifying the main challenges involved in achieving robust linguistic generalization.

## 2. System Overview

The Graph Query Compiler (GQC) v1 is a modular framework designed to translate natural language questions into structured graph query intents.

Instead of directly generating executable graph queries, GQC introduces an intermediate representation layer that separates natural language interpretation from query construction. The system first converts a natural language question into a schema-aware structured intent, which is then validated and compiled into an executable graph query.

This architecture follows a structure-first approach, where query semantics are explicitly represented before execution.

### 2.1 Complete Pipeline

The complete GQC v1 pipeline consists of five main stages:

```text
Natural Language Question
          |
          v
1. Schema-Constrained Intent Prediction
          |
          v
2. Structured Query Intent (JSON)
          |
          v
3. Semantic Validation
          |
          v
4. Query Compilation
          |
          v
5. Graph Query Execution
```

Each stage has a clearly defined responsibility:

#### 1. Schema-Constrained Intent Prediction

The language model receives a natural language question and predicts a schema-constrained structured query intent. The model is responsible for understanding the linguistic meaning and mapping expressions to schema concepts.

#### 2. Structured Query Representation

The predicted output is represented as a structured JSON schema containing the main components required to describe the query:

- target entity;
- relationship paths;
- filters;
- aggregation operations;
- ordering;
- limits;
- returned attributes.

This intermediate representation acts as a bridge between natural language and executable graph queries.

#### 3. Semantic Validation

Before compilation, the generated JSON is validated against structural and semantic constraints.

The validation layer checks:

- schema compatibility;
- valid attributes and operators;
- relationship consistency;
- aggregation validity;
- query regime constraints.

Invalid representations can be identified before reaching the execution layer.

#### 4. Query Compilation

After validation, the structured representation is transformed into an executable graph query language, graph query languages such as Cypher.

The compiler deterministically maps:

- targets → graph nodes;
- paths → relationships;
- filters → conditions;
- aggregations → aggregation clauses;
- ordering and limits → query modifiers.

#### 5. Graph Database Execution

The compiled query is executed against the graph database, producing the final result.

The separation between these stages enables GQC v1 to combine the flexibility of language models with the reliability of explicit schema constraints and deterministic query generation.

### 2.2 System Components

The GQC v1 architecture is composed of the following components:

#### Schema Definition

The schema defines the available graph entities, attributes, relationships, and valid query structures.

It provides the constraints required for intent generation, validation, and query compilation.

#### Intent Generation and Dataset Construction

The dataset generation pipeline creates structured query intents from a predefined graph schema.

This component uses:

- structural regimes;
- field-level policies;
- semantic constraints.

Structural regimes define supported query patterns and complexity levels, while field-level policies control valid combinations of attributes, operators, and values.

The generated intents are transformed into natural language examples, creating supervised data for model adaptation.

#### Model Adaptation

The adapted language model learns the mapping between natural language questions and structured query representations.

The model does not directly generate executable graph queries. Instead, it predicts an intermediate representation that can be validated and compiled by downstream components.

#### Semantic Validation Layer

The validator ensures that predicted representations comply with the defined schema and semantic rules.

It operates independently from the language model, providing an explicit control mechanism over generated outputs.

#### Query Compiler

The compiler transforms validated JSON representations into executable graph queries.

Query compilation is deterministic and separated from language generation, ensuring that execution logic remains independent from model predictions.

### 2.3 Natural Language to JSON Flow

The central operation of GQC v1 is the transformation:

```text
User Question
      |
      v
Natural Language Interpretation
      |
      v
Schema-Constrained Intent Generation
      |
      v
Structured Intent JSON
```

For example:

Natural language question:

```text
"Quais petshops possuem nota acima de 4 no bairro Centro?"
```

The model identifies:

```json
{
  "regime": "relational_lookup_query",
  "target": {
    "label": "Place"
  },
  "filters": [
    {
      "attribute": "rating",
      "node_label": "Place",
      "operator": ">",
      "value": 4
    },
    {
      "attribute": "type",
      "node_label": "Place",
      "operator": "=",
      "value": "pet_store"
    },
    {
      "attribute": "name",
      "node_label": "Neighborhood",
      "operator": "=",
      "value": "Centro"
    }
  ],
  "path": [
    {
      "from": "Place",
      "relationship": "LOCATED_IN",
      "to": "Neighborhood"
    }
  ],
  "return_attributes": [
    "name"
  ]
}
```

This representation preserves the semantic structure of the question while remaining independent from the final query language.

The separation between natural language understanding, structured representation, validation, and compilation allows GQC v1 to provide a more interpretable and controllable approach than end-to-end text-to-query generation.

## 3. Schema Design

The schema design defines the structured representation used by GQC v1 to describe graph queries.

Unlike direct text-to-query approaches, GQC introduces an intermediate schema-aware representation that explicitly models query intent, entities, relationships, constraints, and output requirements.

The schema acts as a contract between the language model, validation layer, and query compiler, ensuring that generated representations remain compatible with the graph structure and execution rules.

### 3.1 Schema Structure

The GQC v1 schema represents graph query intents through a collection of structured fields:

```text
Query Intent
     |
     +-- regime
     |
     +-- target
     |
     +-- filters
     |
     +-- path
     |
     +-- aggregate
     |
     +-- order_by
     |
     +-- limit
     |
     +-- return_attributes
```

Each field has a specific role in describing the intended query operation.

The schema separates:

- **query semantics**, represented by regimes, filters, aggregations, and ordering;
- **graph structure**, represented by targets and relationship paths;
- **output requirements**, represented by returned attributes and limits.

This separation allows the system to validate query intent before compilation.

### 3.2 Query Regimes

GQC v1 defines a set of structural regimes that represent different query patterns and complexity levels.

Regimes determine the expected combination of fields and the required graph operations.

The supported regimes are:

| Regime                         | Description                                                      |
| ------------------------------ | ---------------------------------------------------------------- |
| `simple_lookup_query`          | Retrieves entities based on attributes from a single node        |
| `simple_count_query`           | Counts entities from a single node                               |
| `simple_aggregation_query`     | Applies aggregation functions over attributes from a single node |
| `simple_ranking_query`         | Returns ordered entities from a single node                      |
| `relational_lookup_query`      | Retrieves entities using graph relationships                     |
| `relational_count_query`       | Counts entities across graph relationships                       |
| `relational_aggregation_query` | Applies aggregation over related entities                        |
| `relational_ranking_query`     | Returns ordered entities using graph relationships               |

Simple regimes operate without relationship traversal, while relational regimes require at least one graph path.

### 3.3 Schema Fields

The main schema fields are:

`regime`

Defines the structural query pattern and determines valid field combinations.

`target`

Defines the main graph entity returned by the query.

Example:

```json
{
  "label": "Place"
}
```

`filters`

Defines constraints applied to graph entities.

Each filter specifies:

- attribute;
- node label;
- operator;
- value.

Example:

```json
{
  "attribute": "rating",
  "node_label": "Place",
  "operator": ">",
  "value": 4
}
```

`path`

Defines graph traversal between entities.

Example:

```json
[
  {
    "from": "Place",
    "relationship": "LOCATED_IN",
    "to": "Neighborhood"
  }
]
```

`aggregate`

Defines aggregation operations such as minimum, maximum, average, or count.

Example:

```json
{
  "attribute": "rating",
  "function": "avg"
}
```

`order_by`

Defines ordering operations for ranking queries.

Example:

```json
{
  "attribute": "rating",
  "direction": "desc"
}
```

`limit`

Restricts the number of returned entities.

Example:

```json
{
  "value": 10
}
```

`return_attributes`

Defines the attributes returned after execution.

Example:

```json
[
  "name"
]
```

### 3.4 Complete Schema Example

Natural language question:

```text
"Quais petshops possuem nota acima de 4 no bairro Centro?"
```

Structured query intent:

```json
{
  "regime": "relational_lookup_query",
  "target": {
    "label": "Place"
  },
  "filters": [
    {
      "attribute": "rating",
      "node_label": "Place",
      "operator": ">",
      "value": 4
    },
    {
      "attribute": "type",
      "node_label": "Place",
      "operator": "=",
      "value": "pet_store"
    },
    {
      "attribute": "name",
      "node_label": "Neighborhood",
      "operator": "=",
      "value": "Centro"
    }
  ],
  "path": [
    {
      "from": "Place",
      "relationship": "LOCATED_IN",
      "to": "Neighborhood"
    }
  ],
  "return_attributes": [
    "name"
  ]
}
```

This schema representation captures the complete semantic structure of the query while remaining independent from the underlying graph query language.

By enforcing explicit structural rules, GQC v1 transforms query generation from unconstrained text generation into a schema-guided prediction problem.

## 4. Methodology

This section describes the core components of the Graph Query Compiler (GQC), focusing on how structured query representations are generated, validated, and transformed into executable queries. The methodology is designed to maintain schema consistency and to control the combinatorial space of possible queries, aiming to balance diversity with structural correctness.

### 4.1 Intent Representation

In GQC, queries are represented as structured intents that encode the main components of a graph query in an explicit way. An intent is a schema-aware representation composed of entities, attributes, relationships, and filtering conditions.

This intermediate representation serves as a bridge between natural language and executable queries. By separating query structure from both linguistic expression and execution syntax, the system enables more controlled manipulation, validation, and transformation of query logic.

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

This representation separates high-level intent semantics from schema-specific query structure. The `intent` field encodes the type of operation and its modifiers, while `schema_spec` defines the structural components of the query, including the target entity, filtering conditions, and returned attributes. This separation makes it easier to reason about, validate, and transform query representations while keeping them aligned with the underlying schema.

### 4.2 Intent Generation

#### Structural Foundation

Intent generation in GQC is performed through a schema-driven combinatorial pipeline over a predefined graph schema, where nodes, attributes, and relationships define the space of valid query structures.

The objective is to construct structured query representations (`intent` + `schema_spec`) through controlled combinations of traversal paths and structural components, while maintaining both diversity and structural consistency.

To facilitate controlled experimentation, the generation process is conducted over a simplified graph schema. This allows the evaluation of the core pipeline—particularly intent generation, validation, and compilation—without interference from domain-specific complexities. The framework is designed with the expectation that it can be extended to richer schemas in future iterations.

#### Combinatorial Construction

The generation process explores graph traversal paths with varying depths and composes operators such as filtering, aggregation, and ordering.

Given this schema, candidate intents are constructed by combining:

* target nodes
* attribute-based filters
* relationship paths
* operators such as ordering, limits, and aggregation

#### Structural Regimes (Complexity Control)

To organize the generation process, intents are produced with respect to predefined structural regimes, where each regime defines a class of query patterns with a specific level of complexity. These regimes capture patterns such as simple lookups, multi-attribute filtering, relational queries, and aggregation-based queries.

Each regime implicitly determines which components are present in the query representation, such as filters, relational paths, or aggregation operators. In this way, regimes not only categorize query patterns but also provide constraints for how intents are composed.

By generating intents within these regimes, the system can better control the complexity distribution of the resulting queries, helping to balance coverage across different difficulty levels and reducing bias toward simpler or more frequent patterns.

#### Field-level Policies (Fine-grained Control)

Within each regime, the generation process is guided by field-level policies defined over both the intent and schema specifications. These policies regulate how individual components—such as attributes, operators, and values—can be combined, as well as which values and operators are considered valid in a given context, helping to maintain semantic coherence and avoid unrealistic query patterns.

Operating at the level of individual schema components, these policies provide fine-grained control over how attributes, operators, and values are assigned within each part of the query representation. This helps reduce the combinatorial space while preserving meaningful diversity in the generated queries.

This design introduces multiple levels of control over the generation process, ranging from high-level structural regimes to fine-grained constraints on individual query components.

#### Validation Constraints

To support correctness, generated candidates are checked against structural and semantic constraints, including schema compatibility, type consistency, and limits on query complexity. Candidates that violate these constraints are filtered out before downstream use.

#### Diversity and Coverage

To encourage diversity, a balancing strategy is applied during generation to reduce overrepresentation of specific query patterns, aiming to improve coverage across the query space.

#### Summary

Unlike purely probabilistic approaches, this method constructs query structures within a controlled and deterministic generation space. As a result, generated intents are typically schema-consistent by construction and provide structured coverage for downstream dataset creation and model training.

### 4.3 Semantic Validation

While the intent generation process enforces structural constraints during construction, a dedicated validation layer is used to filter generated intents before they are accepted into downstream stages of the pipeline.

This validation layer operates over the structured intent representation (`intent` + `schema_spec`) and evaluates whether each candidate is structurally consistent, semantically meaningful, compatible with natural language expression, and aligned with domain constraints.

Validation is implemented in a declarative manner using a centralized set of semantic rules, which define domain-level conditions under which entities, attributes, operators, and values are considered valid.

These rules capture constraints such as:

* valid target structures and return attributes
* compatibility between attributes and operators
* semantic consistency of filter definitions
* validity of ordering and aggregation clauses
* coherence of traversal paths
* alignment with attribute semantic types

Importantly, this design separates semantic knowledge from execution logic. The rule set defines *what is considered valid*, while the validator is responsible for applying these rules to filter out invalid or inconsistent query representations.

The validator operates strictly as a verification component. It does not generate, modify, or rank intents, nor does it perform schema traversal. Its role is limited to acting as a gatekeeping stage that determines whether a candidate can proceed through the pipeline.

This separation improves modularity and maintainability, allowing domain-specific constraints to evolve independently from the validation mechanism itself.

By combining schema-level consistency checks with rule-based semantic validation, the system aims to ensure that accepted intents are coherent and suitable for downstream tasks such as dataset construction and model training.

Unlike purely generative approaches, where invalid outputs are often handled post hoc, this design incorporates validation as an explicit filtering stage in the pipeline. As a result, correctness is not assumed by the generation process, but verified before further processing.

### 4.4 Dataset Construction

Once validated intents are obtained, the system constructs a supervised dataset by pairing structured query representations with their corresponding natural language expressions.

Each data sample consists of a tuple:

- natural language query
- structured intent representation (`intent` + `schema_spec`)

This pairing establishes a direct mapping between linguistic input and structured query semantics, enabling the training of models for natural language to query translation.

Natural language expressions are generated from the underlying intent representation, ensuring that each query reflects the structure and semantics encoded in the corresponding schema specification. This helps preserve alignment between form (language) and meaning (query structure), which is important for downstream learning tasks.

The quality of the resulting dataset is strongly influenced by the upstream generation and validation stages. Since intents are constructed within a schema-constrained space, guided by structural regimes and field-level policies, and filtered through a semantic validation layer, the dataset tends to be more consistent and structurally coherent than unconstrained generation approaches.

In addition, a semantic balancing strategy is applied during intent generation to encourage a more diverse distribution of query patterns. This helps reduce overrepresentation of simpler structures and improves coverage across different query types and complexity levels.

As a result, the constructed dataset exhibits:

* structural diversity across query patterns
* balanced distribution of complexity levels
* alignment between natural language and structured queries
* reduction of invalid or incoherent samples

Compared to datasets derived from raw text or weak supervision, this approach focuses on generating data with explicit structural grounding, reducing reliance on noisy annotations.

This dataset serves as the basis for model adaptation, enabling language models to learn mappings from natural language inputs to structured query representations.

### 4.5 Model Adaptation

The dataset constructed from validated intents is used to adapt language models to the task of mapping natural language queries to structured query representations.

This adaptation is performed through supervised fine-tuning, where the model learns to predict structured intents (`intent` + `schema_spec`given a natural language input. The objective is to enable schema-aware inference from user queries, bridging unstructured language and executable query logic.

During training, each sample provides a direct alignment between a natural language expression and its corresponding structured representation. This supervision allows the model to learn compositional mappings between linguistic patterns and query components, such as entities, attributes, filters, and operators.

Unlike standard language modeling, which relies on implicit pattern learning, this formulation defines a structured prediction task where correctness can be evaluated at the level of the generated intent, rather than only at the surface text level.

At inference time, the fine-tuned model receives a natural language query and predicts the corresponding structured representation. This predicted intent serves as an intermediate representation that can be validated and subsequently compiled into an executable query.

The use of structured representations provides several advantages:

* **Interpretability**: The predicted intent can be inspected before execution.
* **Controllability**: Errors can be identified at the structural level.
* **Modularity**: The system separates language understanding from query execution.
* **Robustness**: The model is constrained by the schema, reducing some classes of invalid outputs.

The quality of the model’s performance is influenced by the dataset produced by the GQC pipeline, which emphasizes structural consistency and balanced coverage. However, generalization quality may still depend on linguistic diversity and the complexity of natural language variations in the dataset.

As a result, the adapted model is expected to perform structured natural language to query translation within the constraints defined by the schema and training distribution.

### 4.6 Query Compilation

In the final stage of the pipeline, structured query representations are transformed into executable graph queries. This process, referred to as query compilation, maps the inferred intent (`intent` + `schema_spec`) into a concrete query language such as Cypher.

The compilation process operates deterministically over the structured representation, translating each component of the schema specification into its corresponding query construct. This includes:

* mapping target nodes to `MATCH` clauses
* translating relationship paths into graph traversals
* converting filter conditions into `WHERE` clauses
* applying ordering and limits where specified
* handling aggregation operations when present

Since the input representation has already been validated and constrained by the previous stages, the compilation step does not require ambiguity resolution or additional semantic inference. Instead, it follows a rule-based transformation process designed to preserve structural correctness and reproducibility.

This design separates query generation from query execution. Rather than relying on language models to directly produce executable queries, the system delegates this responsibility to a deterministic compiler operating over a structured intermediate representation.

This separation provides several advantages:

* **Reliability**: The compilation process reduces the likelihood of syntactically invalid queries under the defined schema assumptions.
* **Transparency**: The mapping from intent to query can be inspected and reasoned about.
* **Maintainability**: Adjustments to schema or target query language can be handled at the compiler level.
* **Error Isolation**: Issues can be traced more clearly to either the generation or compilation stages.

Furthermore, this approach reduces dependence on free-form language generation for producing executable queries, helping to avoid common inconsistencies observed in direct text-to-query generation approaches.

As a result, the system produces executable queries derived from structured and validated representations, completing the pipeline from natural language input to graph query execution.

## 5. Demonstration (Current System Behavior)

This section presents a set of representative examples illustrating the current behavior of the Graph Query Compiler (GQC). The objective is not to showcase ideal performance, but to provide a transparent view of how the system operates in practice, including successful predictions, partially correct outputs, and failure cases.

Each example includes:

* a natural language query
* the predicted structured intent
* the compiled query
* a brief analysis of the outcome

These examples reflect the current state of the system and are not intended to provide exhaustive coverage or serve as a formal benchmark.

### 5.1 Successful Cases

#### Example 1 — Simple Filtering

**Natural Language Query:**

```text
Which neighborhoods have an average monthly income greater than 1000 (currency units)?
```

**Predicted Intent:**

```json
{
  "user_intent": "retrieve",
  "schema": {
    "aggregate": null,
    "filters": [
      {
        "attribute": "average_monthly_income",
        "node_label": "Neighborhood",
        "operator": ">",
        "value_float": null,
        "value_int": 1000,
        "value_str": null
      }
    ],
    "limit": null,
    "order_by": null,
    "path": [],
    "return_attributes": [
      "name"
    ],
    "target": {
      "label": "Neighborhood"
    }
  }
}
```

**Generated Cypher:**

```cypher
MATCH (n:Neighborhood)
WHERE n.average_monthly_income > 1000
RETURN n.name AS name
```

**Observation:**

The model accurately maps the natural language query to the structured intent, accurately identifying the target entity (`Neighborhood`), the filtering condition (`average_monthly_income > 1000`), and the return attribute. This example reflects a case where the input closely matches patterns seen during training, resulting in a fully correct prediction and valid query compilation.

#### Example 2 — Multi-attribute Filtering

**Natural Language Query:**

```text
Which places have a rating above 3 and more than 10 reviews?
```

**Predicted Intent:**

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
        "value_int": 3,
        "value_str": null
      },
      {
        "attribute": "num_reviews",
        "node_label": "Place",
        "operator": ">",
        "value_float": null,
        "value_int": 10,
        "value_str": null
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

**Generated Cypher:**

```cypher
MATCH (p:Place)
WHERE p.rating > 3 AND p.num_reviews > 10
RETURN p.name AS name
```

**Observation:**

The model correctly composes multiple filtering conditions from the natural language query, mapping both constraints (`rating > 3` and `num_reviews > 10`) into the structured intent. This demonstrates the model’s ability to handle conjunctions and multi-attribute queries when expressed in familiar linguistic patterns, resulting in a valid and accurate query compilation.

#### Example 3 — Filtering with Ordering and Limit

**Natural Language Query:**

```text
List 10 pet stores with a rating above 4 in ascending order.
```

**Predicted Intent:**

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
        "value_str": "pet_store"
      }
    ],
    "limit": 10,
    "order_by": {
      "attribute": "rating",
      "node_label": "Place"
    },
    "path": [],
    "return_attributes": [
      "name",
      "rating"
    ],
    "target": {
      "label": "Place"
    }
  }
}
```

**Generated Cypher:**

```cypher
MATCH (p:Place)
WHERE p.rating > 4 AND p.type = 'pet_store'
RETURN p.name AS name, p.rating AS rating
ORDER BY p.rating ASC
LIMIT 10
```

**Observation:**

This example demonstrates the model’s ability to compose multiple query components, including filtering, ordering, and limiting. The model correctly captures both attribute constraints (`rating > 4` and `type = pet_store`) and control operators (`ORDER BY`, `LIMIT`), resulting in a structurally complete and executable query. This behavior is observed when the input aligns closely with patterns seen during training.

**Summary**

Across these examples, the model demonstrates strong performance in scenarios where query structure and linguistic patterns closely match those observed during training. In such cases, the model is able to accurately compose filtering conditions and control operators, producing structurally consistent and executable queries.

### 5.2 Partially Correct Cases

#### Example 4 — Incorrect Aggregation Usage

**Natural Language Query:**

```text
Which places have at least 10 reviews?
```

**Predicted Intent:**

```json
{
  "user_intent": "retrieve",
  "schema": {
    "aggregate": {
      "attribute": "num_reviews",
      "function": "count"
    },
    "filters": [
      {
        "attribute": "num_reviews",
        "node_label": "Place",
        "operator": ">=",
        "value_float": null,
        "value_int": 10,
        "value_str": null
      }
    ],
    "limit": null,
    "order_by": null,
    "path": [],
    "return_attributes": [],
    "target": {
      "label": "Place"
    }
  }
}
```

**Generated Cypher:**

```cypher
MATCH (p:Place)
WHERE p.num_reviews >= 10
RETURN p
```

**Observation:**

The model correctly identifies the filtering condition (`num_reviews >= 10`), demonstrating proper understanding of the main constraint. However, it introduces an unnecessary aggregation component (`count(num_reviews)`) that is not required for this query. Additionally, the absence of explicit return attributes reduces clarity in the output structure. This indicates partial understanding of the query intent, with inconsistencies in how structural components are applied.

#### Example 5 — Missing Relationship Path and Invalid Filter Mapping

**Natural Language Query:**

```text
How many pet shops are located in the Campestre neighborhood?
```

**Predicted Intent:**

```json
{
  "user_intent": "retrieve",
  "schema": {
    "aggregate": {
      "attribute": null,
      "function": "count"
    },
    "filters": [
      {
        "attribute": "type",
        "node_label": "Neighborhood",
        "operator": "=",
        "value_float": null,
        "value_int": null,
        "value_str": "Campestre"
      },
      {
        "attribute": "type",
        "node_label": "Place",
        "operator": "=",
        "value_float": null,
        "value_int": null,
        "value_str": "pet_store"
      }
    ],
    "limit": null,
    "order_by": null,
    "path": [],
    "return_attributes": [],
    "target": {
      "label": "Place"
    }
  }
}
```

**Generated Cypher:**

```cypher
// No query generated
```

**Observation:**

The model correctly captures the aggregation intent (`count`) and captures the two main constraints: filtering by place type (`pet_store`) and by neighborhood (`Campestre`). However, it incorrectly maps the neighborhood constraint as an attribute filter instead of modeling it as a relationship traversal. Additionally, the absence of a defined path between `Place` and `Neighborhood` prevents the query from being compiled into Cypher, resulting in no generated output. This indicates partial understanding of the query intent, with limitations in handling relational structures within the schema.

#### Example 6 — Misinterpretation of Superlative Query

**Natural Language Query:**

```text
Which neighborhood has the highest population?
```

**Predicted Intent:**

```json
{
  "user_intent": "retrieve",
  "schema": {
    "aggregate": null,
    "filters": [
      {
        "attribute": "total_resident_population",
        "node_label": "Neighborhood",
        "operator": ">",
        "value_float": null,
        "value_int": 5000,
        "value_str": null
      }
    ],
    "limit": null,
    "order_by": null,
    "path": [],
    "return_attributes": [
      "name"
    ],
    "target": {
      "label": "Neighborhood"
    }
  }
}
```

**Generated Cypher:**

```cypher
MATCH (n:Neighborhood)
WHERE n.total_resident_population > 5000
RETURN n.name AS name
```

**Observation:**

The model correctly identifies the relevant attribute (`total_resident_population`) and applies a filtering condition. However, it misinterprets the superlative expression “highest” as a fixed threshold (`> 5000`) instead of modeling it as an ordering problem (e.g., sorting by population in descending order with a limit of 1). This indicates a limitation in handling comparative and superlative language, where ranking semantics are not properly captured.

**Summary**

Across these examples, the model demonstrates partial understanding of query structure, correctly identifying key components such as filters and aggregation intent. However, consistent limitations are observed in structural composition, relational reasoning, and handling of linguistic nuances such as superlatives. These patterns indicate that while the model captures local aspects of the query, it struggles to fully represent the complete intent in a structurally consistent manner.

### 5.3 Failure Cases

#### Example 7 — Failure to Compose Ranking and Relational Constraints

**Natural Language Query**

```text
Which are the best veterinary clinics in the Centro neighborhood?
```

**Predicted Intent:**

```json
// No intent generated
```

**Generated Cypher:**

```cypher
// No query generated
```

**Observation**

The model fails to produce any structured output for this query, generating neither a valid intent nor a compiled Cypher query. This behavior suggests a breakdown in the inference process when handling composite queries that combine ranking semantics (“best”) with relational constraints (“Centro neighborhood”). While limited linguistic coverage may contribute, the failure primarily reflects an inability to map compositional semantic requirements into a coherent structured representation. This represents a critical failure mode, as the system is unable to recover even a partial interpretation of the query.

#### Example 8 — Attribute Misalignment

**Natural Language Query**

```text
Which places have good service?
```

**Predicted Intent:**

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
        "value_int": 3,
        "value_str": null
      },
      {
        "attribute": "type",
        "node_label": "Place",
        "operator": "=",
        "value_float": null,
        "value_int": null,
        "value_str": "pet_store"
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

**Generated Cypher:**

```cypher
MATCH (p:Place)
WHERE p.rating > 3 AND p.type = 'pet_store'
RETURN p.name AS name
```

**Observation**

The model attempts to interpret the vague expression “good service” by mapping it to a proxy attribute (`rating > 3`), despite the absence of an explicit schema-defined representation for service quality. Additionally, it introduces an unrelated constraint (`type = pet_store`), which is not present in the original query. This results in a semantically incorrect and overly constrained query. The example highlights a limitation in handling abstract or subjective language, where the model relies on heuristic associations rather than grounded schema mappings, leading to attribute misalignment and unintended constraints.

#### Example 9 — Failure to Handle Relational Constraint

**Natural Language Query**

```text
Which pet stores are located in the Jardim neighborhood?
```

**Predicted Intent:**

```json
// No intent generated
```

**Generated Cypher:**

```cypher
// No query generated
```

**Observation**

The model fails to generate any structured output for this query, producing neither an intent nor a Cypher query. This failure occurs in a scenario that requires a relatively simple relational constraint—linking `Place` to `Neighborhood`. Despite the straightforward nature of the query, the model is unable to construct the necessary relationship path or represent the location constraint. This suggests a limitation in handling relational patterns, indicating weak generalization and lack of robustness beyond familiar training distributions.

#### Example 10 — Failure to Handle Relational Filtering with Attribute Constraint

**Natural Language Query**

```text
Which neighborhoods have places with ratings greater than 4?
```

**Predicted Intent:**

```json
// No intent generated
```

**Generated Cypher:**

```cypher
// No query generated
```

**Observation**

The model fails to generate any structured output for a query that requires combining a relational constraint with an attribute-based filter. Specifically, the query involves identifying `Neighborhood` nodes based on a condition applied to related `Place` nodes (`rating > 4`), which requires constructing a traversal path and applying a filter on a connected entity.

Although the underlying schema and training data include relationships between `Neighborhood` and `Place`, these relations are not consistently reflected in the natural language expressions generated during dataset construction. As a result, the model has limited exposure to such relational patterns in linguistic form, leading to failures during inference. This suggests that the issue is not solely due to model limitations, but also to insufficient alignment between structured intents and their corresponding natural language representations.

Overall, this highlights a limitation in handling multi-hop relational reasoning, particularly when such patterns are underrepresented in the training data at the language level.

**Summary**

The failure cases reveal key limitations in the current system. The model exhibits instability during inference, in some cases failing to generate any structured output when queries require the composition of multiple semantic components, such as ranking and relational constraints. Even when outputs are produced, weaknesses in semantic grounding lead to heuristic attribute mappings and the introduction of unsupported constraints, particularly for abstract expressions like “good service.”

The model also struggles with relational reasoning, often failing to construct traversal paths or represent cross-entity constraints, even in relatively simple scenarios. These issues are further amplified when multiple components must be combined within a single query.

Importantly, these failures are not solely due to model limitations, but are also influenced by gaps in the dataset construction process, where structural patterns in the schema are not consistently reflected in natural language. As a result, the model has limited exposure to key compositional and relational patterns during training.

Overall, the results indicate that the primary challenge lies in reliably translating compositional semantic requirements into coherent, schema-aligned structured representations under conditions of limited linguistic and structural coverage.

## 6. Discussion

### 6.0 Experimental Status

The current implementation of the GQC framework is still under active development and has not yet been subjected to a comprehensive empirical evaluation.

Preliminary qualitative experiments and controlled tests suggest that the system can generate structurally consistent query representations under constrained conditions. In particular, the combination of schema-driven generation and semantic validation appears to improve the consistency and validity of intermediate structured outputs.

However, the system still presents limitations in robustness, especially in the model adaptation stage. The model tends to be sensitive to the linguistic patterns seen during training and may struggle with paraphrased or semantically equivalent queries expressed in different forms.

These observations indicate that while the structural aspects of the pipeline are relatively stable under controlled settings, generalization across diverse natural language expressions remains an open challenge.

This work should therefore be understood as an early-stage implementation of the proposed framework, focused on design, decomposition, and controlled experimentation. Future work will focus on expanding empirical evaluation, improving dataset diversity, and systematically analyzing model generalization and failure cases.

### 6.1 Observed Strengths of the Model

The proposed approach shows consistent behavior in scenarios where input queries align with patterns present in the training distribution, particularly in generating structured representations that reflect the underlying schema.

The following strengths were observed:

1. **Structural Consistency**  
The model generally produces structurally coherent outputs that adhere to the defined graph schema. Generated intents correctly capture target nodes, attributes, and filtering operations, suggesting that the schema-driven generation process and validation layer contribute positively to output consistency.

2. **Stability Within the Defined Domain**  
The model exhibits relatively stable behavior across query types supported by the schema. This includes handling basic multi-attribute filtering and simple relational patterns when expressed in familiar formulations, indicating reasonable robustness within the constrained domain.

3. **Learning of Frequent Patterns**  
When input queries follow patterns that are well represented in the training data, the model tends to produce correct structured mappings. This suggests that the model effectively learns correlations between recurring linguistic patterns and their corresponding structured representations.

4. **Limited Structural Compositionality**  
In some cases, the model is able to combine known structural components—such as filters and operators—in slightly novel configurations. However, this behavior is not fully consistent and tends to degrade when linguistic variation increases, indicating partial structural generalization but limited linguistic robustness.

5. **Improved Output Validity via Constrained Pipeline**  
The presence of schema constraints and validation steps in the pipeline contributes to reducing invalid or inconsistent outputs compared to unconstrained generation approaches. This suggests that explicit structure enforcement is beneficial for maintaining output reliability.

### 6.2 Challenges in Intent Dataset Generation

Constructing high-quality datasets for mapping natural language queries to structured graph representations introduces several fundamental challenges. These challenges emerge from the combinatorial nature of query structures, the need for semantic coherence, and the difficulty of aligning structured representations with natural language expressions in a consistent way.

1. **Combinatorial Explosion of Query Structures**  
The space of possible query structures grows rapidly as the number of nodes, attributes, relationships, and operators increases. As traversal depth and structural complexity expand, the number of valid combinations becomes difficult to enumerate exhaustively. This makes naive generation impractical and motivates the use of constrained or guided generation strategies.

2. **Structural vs. Semantic Validity**  
A key challenge lies in distinguishing between structural correctness and semantic plausibility. A query may be valid with respect to the schema—using correct nodes, attributes, and operators—while still representing an unrealistic or unlikely user intent. This requires additional mechanisms beyond schema constraints to filter or guide generation toward more meaningful query patterns.

3. **Distribution Imbalance**  
Uncontrolled or partially controlled generation processes tend to produce an uneven distribution of query types. Simpler or more frequent patterns often dominate the dataset, while more complex or rare structures are underrepresented. This imbalance can influence model behavior during training, leading to bias toward more common query patterns.

4. **Natural Language Alignment**  
Aligning structured query representations with natural language remains inherently ambiguous. A single structured intent can correspond to multiple valid natural language expressions, and conversely, similar expressions may map to different underlying structures. Ensuring consistency in this mapping is a non-trivial aspect of dataset construction.

5. **Lack of High-quality Ground Truth Data**  
Reliable ground truth pairs of natural language and structured queries are difficult to obtain at scale. Manual annotation is expensive and error-prone, especially for complex schemas. As a result, many approaches rely on synthetic or automatically generated data, which may introduce noise or inconsistencies that affect downstream learning quality.

Overall, these challenges highlight the need for controlled, schema-aware, and semantically guided dataset construction approaches, as explored in the GQC framework.

### 6.3 Limitations

Despite the advantages of the proposed approach, several limitations are currently present in the system.

1. **Dependence on Graph Schema**  
The system depends on the availability of a well-defined graph schema. The quality and expressiveness of generated queries are directly tied to the completeness and correctness of this schema. In cases where the schema is incomplete or poorly specified, the quality of generated outputs may be affected.

2. **Manual Definition of Regimes and Policies**  
The framework relies on manually defined structural regimes and field-level policies. While this enables precise control over the generation process, it also introduces additional design effort and may reduce flexibility when adapting the system to new domains.

3. **Closed-domain Assumption**  
The current implementation operates under a closed-domain assumption, where all query structures are constrained by a predefined schema. As a result, the system is not designed to handle open-domain queries or unseen entities, attributes, or relationship types.

4. **Limited Linguistic Generalization**  
The model shows reduced robustness when exposed to paraphrased or linguistically diverse inputs. While performance is relatively stable for familiar patterns, generalization to unseen phrasings remains limited, indicating a dependency on the distribution of the training data.

5. **Noise and Bias in Automatically Generated Datasets**  
Although controlled generation improves structural consistency, automatically generated datasets may still introduce noise or unintended biases if generation rules are not sufficiently constrained. This can result in unrealistic or unevenly distributed query patterns, which may impact model behavior during training.

6. **Limited Linguistic Diversity**  
The generated natural language expressions may not fully capture the variability observed in real-world user queries. Without explicit paraphrasing or augmentation strategies, the dataset can remain somewhat limited in linguistic diversity, which may affect generalization in downstream tasks.

### 6.4 Future Work

Several directions can be explored to extend and improve the proposed framework.

1. **Paraphrasing and Linguistic Augmentation**  
Incorporating paraphrasing strategies is a promising direction for increasing linguistic diversity in the dataset. Generating multiple natural language variations for the same structured intent may improve the model’s robustness to different phrasings and reduce sensitivity to specific patterns observed during training.

2. **Automated Policy and Regime Learning**  
Currently, structural regimes and field-level policies are manually defined. Automating or partially learning these constraints from data or observed query distributions could reduce manual effort and improve scalability when adapting the framework to new domains.

3. **Evaluation and Benchmarking**  
A more systematic evaluation against baseline approaches remains an important next step. This includes measuring structural accuracy, robustness to linguistic variation, and correctness of the final executed queries, enabling a clearer assessment of the framework’s effectiveness.

4. **Extension to More Complex Schemas**  
Extending the framework to support richer and more complex graph schemas is a natural direction for future work. This would allow the system to be applied in more realistic scenarios involving deeper relational structures and more diverse query patterns.

5. **Improved Semantic Validation**  
Enhancing the semantic validation layer to capture more nuanced or context-dependent constraints could further improve the quality of generated intents. This may help reduce edge cases where structurally valid queries are still semantically weak or unrealistic.

## 7. Conclusion

This work introduced the Graph Query Compiler (GQC), a modular framework for generating structurally valid and semantically consistent graph queries from natural language. The proposed approach combines schema-driven intent generation, field-level policies, and a semantic validation layer to support controlled construction of structured query representations and corresponding datasets.

The observations from the current implementation suggest that enforcing structural constraints throughout the generation pipeline contributes to more consistent and interpretable query representations. In particular, separating structural construction from linguistic realization helps reduce invalid or inconsistent outputs in controlled settings.

At the same time, the analysis indicates that linguistic generalization remains a limitation, especially when dealing with paraphrased or distributionally different inputs from those seen during training. This highlights the importance of dataset diversity and further improvements in natural language coverage.

Overall, this work represents an initial step toward a controlled, schema-aware approach for natural language to graph query generation. The results suggest that explicit structure and validation mechanisms can improve reliability, while also indicating several directions for future refinement and empirical evaluation.
