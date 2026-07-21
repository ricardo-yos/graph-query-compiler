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

## 4. Schema-Driven Dataset Construction

The GQC v1 dataset construction process is based on a schema-driven synthetic generation approach.

Instead of relying exclusively on manually collected query examples, GQC generates structured query intents from predefined graph structures, semantic rules, and field-level constraints.

The objective is to create supervised data where each natural language question is paired with a validated structured representation.

The generation pipeline follows:

```text
Graph Schema
      |
      v
Structural Regime Selection
      |
      v
Intent Expansion
      |
      v
Semantic Validation
      |
      v
Natural Language Generation
      |
      v
Training Examples
```

This approach allows controlled coverage of different query patterns while preserving consistency between natural language expressions and graph query semantics.

### 4.1 Intent Generation

The generation process starts from the graph schema and expands valid query structures according to predefined rules.

The generator creates structured intents by combining:

- query regimes;
- graph paths;
- target entities;
- filters;
- operators;
- aggregation functions;
- ordering constraints;
- return attributes.

Each generated intent represents a valid query structure before being converted into a natural language question.

This separation allows the system to control the distribution of examples and avoid invalid query combinations.

### 4.2 Structural Regimes

Structural regimes define the supported query patterns and determine which schema fields are applicable.

For example:

- lookup regimes require entity retrieval;
- count regimes require counting operations;
- aggregation regimes require aggregation functions;
- ranking regimes require ordering operations.

The distinction between simple and relational regimes controls graph complexity:

```text
Simple Query
    Entity
      |
      v
    Filter

Relational Query
    Entity
      |
      v
 Relationship Traversal
      |
      v
 Related Entity Constraints
```

Simple regimes operate on a single graph entity, while relational regimes require explicit relationship traversal through graph paths.

### 4.3 Field-Level Policies

Field-level policies control how individual schema components are generated.

They define valid combinations for:

- attributes;
- operators;
- values;
- filters;
- aggregation functions;
- ordering directions.

Examples:

- `rating` supports numerical comparison operators;
- `type` supports categorical equality constraints;
- ranking queries require an ordering attribute;
- aggregation queries require an aggregate function.

These policies reduce invalid combinations and improve semantic consistency in generated examples.

### 4.4 Intent Validation During Generation

Before becoming training data, generated intents are validated against schema and semantic constraints.

The validation stage checks:

- valid regime-field combinations;
- required fields;
- graph path consistency;
- attribute compatibility;
- operator compatibility;
- aggregation rules.

Invalid intents are discarded, ensuring that only structurally valid examples are used for model adaptation.

### 4.5 Natural Language Generation

After validation, structured intents are converted into natural language questions.

The generated questions preserve the semantic meaning of the underlying intent while introducing linguistic variation.

For example:

Structured intent:

```json
{
  "regime": "simple_count_query",
  "filters": [
    {
      "attribute": "type",
      "node_label": "Place",
      "operator": "=",
      "value": "pet_store"
    }
  ]
}
```

Possible natural language expressions:

```text
"Quantos petshops existem?"
```

```text
"Qual é a quantidade de lojas classificadas como petshop?"
```

The semantic structure remains unchanged while the linguistic surface form varies.

### 4.6 Dataset Characteristics

The generated dataset provides:

- controlled structural diversity;
- balanced coverage across query regimes;
- explicit semantic supervision;
- consistent alignment between language and graph structure.

By generating examples from the schema itself, GQC v1 transforms dataset construction into a controlled engineering process rather than a purely data collection problem.

## 5. Model Training

The GQC v1 model training process focuses on adapting a pre-trained language model to generate schema-constrained query intents from natural language questions.

Instead of training a model from scratch, GQC v1 uses parameter-efficient fine-tuning to adapt an existing instruction-tuned language model.

The training objective is to learn the transformation:

```text
Natural Language Question
              |
              v
       Adapted Language Model
              |
              v
    Structured Query Intent (JSON)
```

The model learns to generate valid intermediate representations according to the GQC schema while preserving the semantic meaning of the original question.

### 5.1 Base Model

GQC v1 uses an instruction-tuned large language model as the foundation for adaptation.

The base model provides general language understanding capabilities, while fine-tuning specializes its behavior toward structured query intent generation.

The model is not trained to directly generate graph queries. Instead, it learns the output format and semantic constraints defined by the GQC schema.

The adaptation process leverages the model's existing capabilities for:

- natural language understanding;
- instruction following;
- structured output generation.

Fine-tuning focuses these capabilities on the graph query domain.

### 5.2 Parameter-Efficient Fine-Tuning (LoRA / QLoRA)

GQC v1 uses Parameter-Efficient Fine-Tuning (PEFT) techniques to adapt the base model.

Instead of updating all model parameters, LoRA introduces trainable low-rank matrices while keeping the original model weights frozen.

The optimization objective can be represented as:

```text
Base Model Parameters (Frozen)
              +
      LoRA Adapter Weights
              |
              v
      Adapted GQC Model
```

QLoRA extends this approach by combining LoRA adaptation with quantized model weights, reducing memory requirements while maintaining training effectiveness.

The advantages of this approach include:

- reduced GPU memory consumption;
- faster experimentation;
- preservation of the original model capabilities;
- easier iteration over different dataset versions.

For GQC v1, QLoRA enables experimentation with domain-specific structured generation using consumer-grade hardware.

### 5.3 Training Configuration and Hyperparameters

The training process uses a supervised fine-tuning objective where each example contains:

an instruction or question;
the expected structured JSON output.

The main training parameters include:

| Parameter             | Description                                         |
| --------------------- | --------------------------------------------------- |
| Base model            | Instruction-tuned language model used as foundation |
| Fine-tuning method    | QLoRA                                               |
| Training objective    | Structured intent generation                        |
| Learning rate         | Optimization step size                              |
| Batch size            | Number of examples processed per update             |
| Gradient accumulation | Effective batch size control                        |
| Number of epochs      | Number of dataset passes                            |
| Sequence length       | Maximum input/output token length                   |
| LoRA rank             | Adapter capacity                                    |
| LoRA alpha            | Adapter scaling factor                              |
| Dropout               | Regularization parameter                            |

Hyperparameter selection focuses on balancing:

- schema accuracy;
- JSON generation reliability;
- generalization across query regimes.

### 5.4 Training Prompt Design

The training prompt defines how the model receives instructions and how the expected output format is represented.

The prompt provides:

- the task definition;
- schema constraints;
- expected JSON structure;
- input question;
- output format requirements.

Example:

```text
You are a graph query intent generation model.

Given a natural language question, generate a valid JSON representation according to the GQC schema.

The output must contain:
- regime;
- target;
- filters;
- path;
- aggregate;
- order_by;
- limit;
- return_attributes.

Question:

{user_question}

Output:

{json_schema}
```

The prompt design aims to reduce ambiguity by explicitly describing the expected intermediate representation.

The model is therefore optimized for structured semantic prediction rather than unrestricted text generation.

### 5.5 Training Objective

The training objective is supervised next-token prediction over the structured intent representation.

Given an input question:

$$
x = \text{input natural language question}
$$

$$
y = \text{structured query intent JSON}
$$

The model learns the conditional distribution:

$$
P(y|x)
$$

where:

- \(x\) represents the natural language question;
- \(y\) represents the structured JSON intent.

The objective is to maximize the probability of generating the correct schema representation while maintaining semantic consistency with the input question.

This training strategy aligns the language model with the GQC architecture, where generation is constrained by explicit schema definitions.
