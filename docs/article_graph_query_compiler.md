# Graph Query Compiler: Structured Reasoning for Reliable Natural Language Interfaces to Knowledge Graphs

## 1. Introduction

Large language models (LLMs) have demonstrated strong capabilities in generating fluent and contextually coherent text. However, their ability to perform structured reasoning over constrained domains remains limited. Tasks that require consistency with an underlying schema—such as graph query generation—often expose these limitations, as models may produce outputs that are syntactically plausible but semantically invalid. Even when LLMs appear to perform reasoning, this process remains largely implicit and does not reliably enforce domain-specific constraints.

These limitations become critical in scenarios where multiple attributes, relationships, and filtering conditions must be composed while preserving structural validity. In such cases, LLMs frequently generate outputs that violate schema constraints, introduce incompatible attribute combinations, or misalign the intended query structure with its natural language representation.

Approaches that rely solely on language modeling cannot reliably enforce structural constraints. As a result, they often produce inconsistent or semantically invalid outputs, limiting their effectiveness in tasks such as semantic parsing, dataset generation, and natural language interfaces over structured data systems.

To address this gap, we propose the Graph Query Compiler (GQC), a deterministic pipeline that separates structural reasoning from linguistic realization. The system first generates candidate query intents through controlled combinatorial processes and enforces schema consistency via a semantic validation layer, before mapping these validated structures into natural language. By enforcing structural correctness prior to language generation, GQC mitigates common failure modes of LLMs and enables the creation of high-quality, semantically consistent datasets.

This approach provides a more reliable foundation for tasks that depend on structured reasoning, while preserving the expressive flexibility of natural language.
