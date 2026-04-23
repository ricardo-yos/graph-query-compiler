# Graph Query Compiler: Structured Reasoning for Reliable Natural Language Interfaces to Knowledge Graphs

## Abstract

Generating accurate structured queries from natural language remains a challenging problem, particularly in graph-based systems where queries must satisfy both structural and semantic constraints. Existing approaches often rely on large language models or weakly supervised data, which can lead to inconsistencies, invalid query structures, and limited controllability.

In this work, we introduce the Graph Query Compiler (GQC), a modular framework for controlled generation of structured graph queries. The proposed approach decomposes the problem into distinct stages, including schema-driven intent generation, dataset construction, model adaptation, and query compilation. Central to the framework is a combinatorial generation process guided by structural regimes and field-level policies, enabling systematic exploration of valid query structures while limiting combinatorial explosion. A semantic validation layer further enforces structural and semantic consistency.

This work focuses on the design and analysis of a controlled framework for structured query generation, rather than on comprehensive empirical benchmarking. Preliminary qualitative observations suggest that the approach enables the construction of structurally consistent datasets and reliable intermediate representations, while also highlighting limitations in linguistic generalization.

These findings indicate that enforcing structure and control during dataset generation and query construction is a promising direction for improving natural language interfaces to graph databases.

## 1. Introduction

Large language models (LLMs) have demonstrated strong capabilities in generating fluent and contextually coherent text. However, their ability to perform structured reasoning over constrained domains remains limited. Tasks that require consistency with an underlying schema—such as graph query generation—often expose these limitations, as models may produce outputs that are syntactically plausible but semantically invalid. Even when LLMs appear to perform reasoning, this process remains largely implicit and does not reliably enforce domain-specific constraints.

These limitations become critical in scenarios where multiple attributes, relationships, and filtering conditions must be composed while preserving structural validity. In such cases, LLMs frequently generate outputs that violate schema constraints, introduce incompatible attribute combinations, or misalign the intended query structure with its natural language representation.

Approaches that rely solely on language modeling cannot reliably enforce structural constraints. As a result, they often produce inconsistent or semantically invalid outputs, limiting their effectiveness in tasks such as semantic parsing, dataset generation, and natural language interfaces over structured data systems.

To address this gap, we propose the Graph Query Compiler (GQC), a partially deterministic pipeline that separates structural reasoning from linguistic realization. The system first generates candidate query intents through controlled combinatorial processes and enforces schema consistency via a semantic validation layer, before mapping these validated structures into natural language. By enforcing structural correctness prior to language generation, GQC mitigates common failure modes of LLMs and enables the creation of high-quality, semantically consistent datasets.

This approach provides a more reliable foundation for tasks that depend on structured reasoning, while preserving the expressive flexibility of natural language.

This work focuses on the design, formalization, and analysis of a controlled framework for structured query generation, rather than on comprehensive empirical evaluation or state-of-the-art performance. The goal is to establish a reliable foundation for future empirical studies by explicitly addressing structural validity, semantic consistency, and controllability in query construction.

Prior work in semantic parsing and text-to-query generation has explored mapping natural language to structured representations, often relying on neural models or weak supervision. In contrast, this work emphasizes explicit structural control through schema-driven combinatorial generation and rule-based validation, focusing on enforcing correctness prior to language generation.

## 2. Contributions

The main contributions of this work are as follows:

1. **A Schema-driven Framework for Structured Query Generation**  
We propose the Graph Query Compiler (GQC), a modular framework that formulates natural language to graph query translation as a structured, schema-aware generation problem, explicitly separating structural reasoning from linguistic realization.

2. **Controlled Combinatorial Intent Generation**  
We introduce a deterministic, schema-driven approach to intent generation based on structural regimes and field-level policies, enabling systematic exploration of valid query structures while controlling combinatorial complexity.

3. **Multi-level Constraint Mechanism**  
We design a layered constraint system that combines high-level structural regimes with fine-grained field-level policies, providing explicit control over query composition and ensuring structural and semantic coherence.

4. **Semantic Validation as a First-class Component**  
We formalize a rule-based semantic validation layer that operates independently of generation, enforcing domain constraints and filtering invalid or unrealistic query representations prior to downstream use.

5. **A Modular Pipeline for Reliable Query Construction**  
We present a modular architecture that decomposes the process into intent generation, dataset construction, model adaptation, and query compilation, enabling interpretability, controllability, and separation of concerns across stages.

6. **Analysis of Challenges in Intent Dataset Construction**  
We provide a structured analysis of the key challenges involved in generating high-quality datasets for structured query learning, including combinatorial explosion, semantic validity, distribution imbalance, and linguistic alignment.

7. **Preliminary Analysis of Structural vs. Linguistic Generalization**  
Through preliminary observations, we highlight a key limitation of current approaches: while structural generalization can be achieved through controlled generation, linguistic generalization remains a significant challenge, motivating future work on data augmentation and paraphrasing.

## 3. System Overview

The Graph Query Compiler (GQC) is a partially deterministic pipeline that generates and validates structured query representations before compiling them into executable graph queries. By explicitly separating structural reasoning from linguistic realization, the system ensures that all outputs are consistent with a predefined schema and can be reliably executed in graph database environments.

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

## 4. Methodology

This section describes the core components of the Graph Query Compiler (GQC), focusing on how structured query representations are generated, validated, and transformed into executable queries. The methodology is designed to enforce schema consistency and control the combinatorial space of possible queries, ensuring both diversity and structural correctness.

### 4.1 Intent Representation

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

### 4.2 Intent Generation

#### Structural Foundation

Intent generation in GQC is performed through a schema-driven combinatorial pipeline over a predefined graph schema, where nodes, attributes, and relationships define the space of valid query structures.

The objective is to systematically construct structured query representations (`intent` + `schema_spec`) through controlled combinations of traversal paths and structural components, while ensuring both diversity and structural correctness.

To facilitate controlled experimentation, the generation process is conducted over a simplified graph schema. This allows the evaluation of the core pipeline—particularly intent generation, validation, and compilation—without interference from domain-specific complexities. The framework, however, is designed to generalize to richer schemas as future work.

#### Combinatorial Construction

The generation process involves exploring graph traversal paths with varying depths and composing operators such as filtering, aggregation, and ordering.

Given this schema, candidate intents are generated by composing:

* target nodes
* attribute-based filters
* relationship paths
* operators such as ordering, limits, and aggregation

#### Structural Regimes (Complexity Control)

To structure the generation process, intent generation is performed with respect to predefined structural regimes, where each regime defines a class of query patterns with a specific level of query complexity. These regimes capture patterns such as simple lookups, multi-attribute filtering, relational queries, and aggregation-based queries.

Each regime implicitly defines which components are present in the query representation, such as the inclusion of filters, relational paths, or aggregation operators. In this way, regimes not only categorize query patterns but also constrain the composition of generated intents.

By generating intents within these regimes, the system gains explicit control over the complexity of the resulting queries, ensuring balanced coverage across different levels of difficulty and preventing the dataset from being biased toward simpler or more frequent query patterns.

#### Field-level Policies (Fine-grained Control)

Within each regime, the generation process is guided by field-level policies defined over both the intent and schema specifications. These policies regulate how individual components—such as attributes, operators, and values—can be combined, as well as which values and operators are considered valid within a given context, ensuring semantic coherence and eliminating invalid or unrealistic query patterns.

Operating at the level of individual schema components, these policies enable fine-grained control over how attributes, operators, and values are assigned within each part of the query representation. By constraining the admissible combinations at this level, the system effectively limits the combinatorial explosion inherent to the generation process while preserving diversity.

This design introduces multiple levels of control over the generation process, ranging from high-level structural regimes to fine-grained constraints on individual query components.

#### Validation Constraints

To ensure validity, generated candidates are subject to structural and semantic constraints, including schema compatibility, type consistency, and limits on query complexity.

#### Diversity and Coverage

To promote diversity, a semantic balancing strategy is applied to prevent overrepresentation of specific query patterns, ensuring more comprehensive coverage of the query space.

#### Summary

Unlike probabilistic approaches, this method explicitly constructs query structures within a controlled and deterministic search space. As a result, all generated intents are schema-consistent by design and provide high structural coverage for downstream dataset construction and model training.

### 4.3 Semantic Validation

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

### 4.4 Dataset Construction

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

### 4.5 Model Adaptation

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

### 4.6 Query Compilation

In the final stage of the pipeline, structured query representations are transformed into executable graph queries. This process, referred to as query compilation, maps the inferred intent (`intent` + `schema_spec`) into a concrete query language such as Cypher.

The compilation process operates deterministically over the structured representation, translating each component of the schema specification into its corresponding query construct. This includes:

* mapping target nodes to MATCH clauses
* translating relationship paths into graph traversals
* converting filter conditions into WHERE clauses
* applying ordering and limits where specified
* handling aggregation operations when present

Because the input representation is already validated and schema-consistent, the compilation step does not require ambiguity resolution or complex inference. Instead, it follows a rule-based transformation process that ensures correctness and reproducibility.

This design separates query generation from query execution. Rather than relying on language models to directly produce executable queries, the system delegates this responsibility to a deterministic compiler that operates over a structured intermediate representation.

This separation provides several advantages:

* **Reliability**: Generated queries are guaranteed to be syntactically and structurally valid.
* **Transparency**: The transformation from intent to query can be inspected and audited.
* **Maintainability**: Changes to the query language or schema can be handled at the compiler level.
* **Error Isolation**: Failures can be traced back to either the prediction stage or the compilation stage.

Furthermore, this approach mitigates common failure modes of language models, such as producing invalid or inconsistent queries, by removing direct dependency on free-form text generation for executable outputs.

As a result, the system ensures that all executable queries are derived from validated, interpretable, and schema-aligned representations, completing the end-to-end pipeline from natural language input to reliable query execution.

## 5. Discussion

### 5.0 Experimental Status

The current implementation of the GQC framework is still under active development and presents limitations that prevent a comprehensive empirical evaluation at this stage.

Preliminary qualitative experiments and controlled observations indicate that the system is capable of generating structurally consistent query representations under controlled conditions. In particular, the combination of schema-driven generation and semantic validation contributes to the production of valid and interpretable structured outputs.

However, the system exhibits limitations in robustness, especially in the model adaptation stage. The learned model remains sensitive to the linguistic patterns observed during training and struggles to generalize to paraphrased or structurally equivalent queries expressed in diverse forms.

These observations motivate the analysis presented in the following sections and highlight key areas for improvement, including dataset diversity, paraphrasing strategies, and refinement of semantic validation. As such, the current work should be understood as a step toward a more complete system, with future work focusing on empirical validation and performance benchmarking.

### 5.1 Observed Strengths of the Model

The proposed approach demonstrates strong performance in scenarios where query structure aligns with patterns observed during training, particularly in generating accurate structured representations from familiar inputs.

The following strengths were observed:

1. **Structural Consistency**  
The model consistently produces structurally coherent outputs that adhere to the underlying graph schema. Generated intents correctly identify target nodes, attribute-based filters, and operators, reflecting the effectiveness of the schema-driven generation process and the validation mechanisms embedded in the pipeline.

2. **Robustness Within the Domain**  
The model maintains stable behavior across a range of query types supported by the schema. This includes handling multi-attribute filtering and basic relational patterns when expressed in familiar linguistic forms, demonstrating robustness within the defined domain.

3. **Accurate Mapping of Learned Patterns**  
When input queries follow known syntactic and semantic patterns, the model reliably maps them to correct structured representations. This indicates strong alignment between the training data and the learned mappings.

4. **Structural Generalization**  
The model is capable of combining known structural components—such as filters and operators—in novel ways. While it may struggle with diverse linguistic expressions, this suggests that the model has learned aspects of the underlying query structure, although its linguistic generalization remains limited.

5. **Reliability Through Controlled Generation**  
The integration of validation and controlled generation contributes to the overall reliability of the system, reducing the likelihood of producing invalid or incoherent query representations.

### 5.2 Challenges in Intent Dataset Generation

Constructing high-quality datasets for mapping natural language queries to structured graph representations presents several fundamental challenges. These challenges arise from the combinatorial nature of query structures, the need for semantic coherence, and the difficulty of aligning structured representations with natural language expressions.

1. **Combinatorial Explosion of Query Structures**  
The combinatorial explosion of possible query structures is one of the primary challenges in dataset construction. Given a graph schema with multiple nodes, attributes, relationships, and operators, the number of possible query combinations grows rapidly as traversal depth and structural complexity increase. Naively enumerating all possible combinations leads to an intractable search space, making controlled generation essential.

2. **Structural vs. Semantic Validity**  
Another key challenge is the distinction between structural and semantic validity. A query may be structurally consistent with the schema—using valid nodes, attributes, and relationships—yet still be semantically meaningless or unrealistic in practice. For example, certain combinations of attributes and operators may not correspond to plausible user intents, requiring an additional layer of semantic validation beyond structural correctness.

3. **Distribution Imbalance**  
Imbalance in the distribution of generated queries also presents a significant challenge. Unconstrained generation tends to favor simpler or more common query patterns, leading to datasets that underrepresent more complex or less frequent structures. This imbalance can negatively impact model training, as the model may become biased toward specific query types.

4. **Natural Language Alignment**  
Aligning structured representations with natural language expressions is inherently challenging. A single structured query may correspond to multiple valid linguistic expressions, while similar natural language queries may map to different underlying structures. Ensuring consistent and meaningful alignment between these representations is critical for supervised learning.

5. **Lack of High-quality Ground Truth Data**  
A key challenge in dataset construction is the lack of reliable ground truth data. Creating accurate pairs of natural language queries and their corresponding structured representations requires manual annotation, which is time-consuming, error-prone, and difficult to scale, especially for complex graph schemas. As a result, many existing approaches rely on automatically generated or weakly supervised data, which may introduce noise and inconsistencies that negatively affect model performance.

These challenges motivate the need for controlled, schema-aware, and semantically grounded dataset construction methods, as realized in the GQC framework.

### 5.3 Limitations

Despite its advantages, the proposed approach presents several limitations.

1. **Dependence on Graph Schema**  
The system relies on the availability of a well-defined graph schema. The quality and expressiveness of the generated queries are directly influenced by the completeness and accuracy of this schema, which may limit the system’s effectiveness if the schema is incomplete or poorly specified.

2. **Manual Definition of Regimes and Policies**  
The use of predefined structural regimes and field-level policies requires manual specification and domain knowledge. While this enables fine-grained control over the generation process, it may introduce additional effort when adapting the framework to new domains.

3. **Closed-domain Assumption**  
The framework operates within a closed-domain setting, where all query structures are constrained by a predefined schema. As a result, it does not generalize to open-domain queries or to scenarios involving unseen entities, attributes, or relationships.

4. **Limited Linguistic Generalization**  
The model shows reduced performance when handling paraphrased or linguistically diverse inputs. Although it generalizes well at the structural level, its reliance on patterns present in the training data can lead to errors when queries are expressed in unfamiliar ways.

5. **Noise and Bias in Automatically Generated Datasets**  
While automatic dataset generation enables scalability and structural consistency, it may introduce noise and biases if not properly constrained. In particular, weak semantic control or inconsistent generation rules can result in invalid, unrealistic, or systematically skewed query patterns, negatively affecting model reliability.

6. **Limited Linguistic Diversity**  
Automatically generated datasets may also lack sufficient linguistic variation. Without explicit mechanisms for generating diverse natural language expressions, the dataset may fail to capture the variability of real-world queries, limiting downstream model generalization.

### 5.4 Future Work

Several directions can be explored to extend and improve the proposed framework.

1. **Paraphrasing and Linguistic Augmentation**  
Incorporating paraphrasing strategies to increase linguistic diversity is a promising direction. Generating multiple natural language variants for each structured intent may improve the model’s ability to generalize to unseen formulations.

2. **Automated Policy and Regime Learning**  
Automating the definition of structural regimes and field-level policies could reduce manual effort and improve scalability. Learning these constraints directly from data or usage patterns is a potential avenue for future research.

3. **Evaluation and Benchmarking**  
Conducting systematic evaluations against baseline approaches is an important next step. This includes measuring structural accuracy, robustness to linguistic variation, and execution correctness.

4. **Extension to More Complex Schemas**  
Expanding the framework to support richer and more complex graph schemas would increase its applicability in real-world scenarios, including domains with deeper relational structures.

5. **Improved Semantic Validation**  
Enhancing the semantic validation layer to capture more nuanced constraints could further improve the quality of generated intents and reduce unrealistic query patterns.

## 6. Conclusion

This work introduced the Graph Query Compiler (GQC), a modular framework for generating structurally valid and semantically consistent graph queries from natural language. By combining schema-driven intent generation, field-level policies, and a dedicated semantic validation layer, the approach enables controlled construction of high-quality intent datasets and reliable query representations.

The results highlight the effectiveness of enforcing structure and constraints throughout the pipeline, leading to consistent and interpretable outputs. At the same time, the analysis reveals that linguistic generalization remains a key challenge, emphasizing the importance of dataset design and diversity.

Overall, this work demonstrates that treating query generation as a controlled, schema-aware process provides a viable path toward more reliable natural language interfaces for graph-based systems.
