"""
Combinatorial Intent Generation Pipeline
=======================================

Builds a structurally diverse dataset of graph query intents by
systematically exploring combinations of paths, filters, projections
and operators defined in a graph schema.

Generation strategy:
- explore graph traversal paths with different depths
- combine structural operators (filters, aggregation, ordering)
- apply semantic validation rules
- enforce diversity using semantic bucket balancing

Pipeline steps:
1. Load global generation configuration
2. Load graph schema
3. Generate intents under different structural regimes
4. Validate structural and semantic correctness
5. Apply semantic diversity balancing
6. Export dataset in JSONL format

Inputs:
- combinatorial.yaml
    global dataset generation policy

- regime_types.yaml
    definitions of structural complexity regimes

- graph_schema.json
    entities, attributes and relations of the graph domain

Outputs:
- structural_intents.jsonl
    structured query intents ready for NL → query training

Purpose:
Provide high structural coverage of possible graph queries,
enabling robust training of semantic parsers or LLMs that map
natural language questions into structured graph queries.
"""

import json
import random
from collections import defaultdict
import yaml

from config.paths import SCHEMA_DATA_DIR, INTENTS_DATA_DIR, INTENTS_CONFIG_DIR

from src.intents.generation.structural_config import StructuralGenerationConfig
from src.intents.generation.graph_schema_adapter import GraphSchemaAdapter
from src.intents.generation.combinatorial_generator import CombinatorialStructuralGenerator

from src.intents.validation.intent_validator import is_valid_intent
from src.intents.dataset_curation.semantic_bucket_selector import SemanticBucketSelector


# ensures reproducibility of stochastic sampling
SEED = 42
random.seed(SEED)


def load_generation_config() -> StructuralGenerationConfig:
    """
    Loads the global combinatorial generation configuration.

    The configuration defines:
    - semantic diversity balancing policy
    - limits on projections and filters
    - constraints controlling combinatorial explosion

    Returns
    -------
    StructuralGenerationConfig
        validated configuration object used by the generator
    """

    config_path = INTENTS_CONFIG_DIR / "combinatorial.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Generation config not found: {config_path}"
        )

    with open(config_path, encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    return StructuralGenerationConfig(**raw_config)


def load_regime_types():
    """
    Loads structural regimes defining query complexity patterns.

    Each regime constrains which operators can be used
    during intent generation.

    Examples of constraints:
    - maximum graph depth
    - aggregation availability
    - ordering availability
    - filter multiplicity

    Returns
    -------
    dict
        mapping regime name → structural constraints
    """

    path = INTENTS_CONFIG_DIR / "regime_types.yaml"

    if not path.exists():
        raise FileNotFoundError(
            f"Regime config not found: {path}"
        )

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return raw["regimes"]


def load_schema():
    """
    Loads graph schema and converts it into an internal adapter.

    The adapter provides a normalized interface for:
    - entities
    - attributes
    - relationships
    - traversal paths

    Returns
    -------
    GraphSchemaAdapter
        abstraction used by the combinatorial generator
    """

    schema_path = SCHEMA_DATA_DIR / "graph_schema.json"

    with open(schema_path, encoding="utf-8") as f:
        raw_schema = json.load(f)

    schema = GraphSchemaAdapter(raw_schema)

    print(f"Schema loaded from:\n{schema_path}")

    return schema


def build_config_from_regime(base_config, regime_name, regime):
    """
    Creates a specialized generation config for a given regime.

    Each regime defines a structural complexity level,
    allowing the generator to produce queries with different
    compositional properties.

    Parameters
    ----------
    base_config : StructuralGenerationConfig
        global configuration template

    regime_name : str
        label identifying the structural regime

    regime : dict
        constraints defining the regime

    Returns
    -------
    StructuralGenerationConfig
        regime-specific configuration
    """

    cfg = base_config.model_copy(deep=True)

    cfg.regime_name = regime_name

    cfg.max_depth = regime["max_depth"]

    cfg.allow_aggregation = regime["allow_aggregation"]

    cfg.allow_order_by = regime["allow_order_by"]

    cfg.allow_limit = regime["allow_limit"]

    cfg.allow_multiple_filters = regime["allow_multiple_filters"]

    cfg.max_filters_per_node = regime["max_filters_per_node"]

    return cfg


def generate_intents_by_regime(schema, base_config):
    """
    Generates intents separately for each structural regime.

    This ensures coverage across multiple levels of query complexity.

    Examples of regimes:
    - simple lookup
    - multi-hop traversal
    - analytical queries with aggregation

    Parameters
    ----------
    schema : GraphSchemaAdapter

    base_config : StructuralGenerationConfig

    Returns
    -------
    list
        generated intent objects
    """

    regimes = load_regime_types()

    all_intents = []

    for regime_name, regime in regimes.items():

        print(f"\nGenerating regime: {regime_name}")

        regime_config = build_config_from_regime(
            base_config,
            regime_name,
            regime
        )

        generator = CombinatorialStructuralGenerator(
            schema=schema,
            config=regime_config
        )

        intents = generator.generate()

        # attach regime label for later balancing
        for intent in intents:
            intent.intent.regime = regime_name

        print(f"Generated: {len(intents)}")

        all_intents.extend(intents)

    print("\nTotal generated:", len(all_intents))

    return all_intents


def filter_valid_intents(intents):
    """
    Removes structurally or semantically invalid intents.

    Validation ensures:
    - schema compatibility
    - semantic correctness
    - allowed operator combinations
    - logical coherence of filters and projections

    Parameters
    ----------
    intents : list

    Returns
    -------
    list
        filtered valid intents
    """

    valid_intents = []

    removed = 0

    for intent in intents:

        intent_dict = intent.model_dump()

        if is_valid_intent(intent_dict):

            valid_intents.append(intent)

        else:

            removed += 1

    print(f"Valid intents   : {len(valid_intents)}")

    print(f"Rejected intents: {removed}")

    return valid_intents


def apply_semantic_balance_by_regime(intents, config, top_k_regimes=3):
    """
    Applies semantic diversity balancing selectively to high-volume regimes.

    Prevents the dataset from being dominated by repetitive
    structural patterns.

    Strategy:
    - group intents by regime
    - identify regimes with largest volume
    - apply bucket-based sampling only to these regimes

    Parameters
    ----------
    intents : list

    config : StructuralGenerationConfig

    top_k_regimes : int
        number of regimes where balancing will be applied

    Returns
    -------
    list
        semantically balanced intents
    """

    if not config.enable_semantic_balance:

        return intents

    grouped = defaultdict(list)

    for intent in intents:

        grouped[intent.intent.regime].append(intent)

    regimes_sorted = sorted(

        grouped.items(),

        key=lambda x: len(x[1]),

        reverse=True

    )

    regimes_to_balance = {

        name for name, _ in regimes_sorted[:top_k_regimes]

    }

    print("Semantic balance applied to regimes:")

    for r in regimes_to_balance:

        print(" -", r)

    selector = SemanticBucketSelector(config)

    final_intents = []

    for regime, group in grouped.items():

        if regime in regimes_to_balance:

            balanced = selector.select(group)

            print(
                f"{regime}: {len(group)} -> {len(balanced)}"
            )

            final_intents.extend(balanced)

        else:

            final_intents.extend(group)

    print(f"\nBefore balance: {len(intents)}")

    print(f"After balance : {len(final_intents)}")

    return final_intents


def export_jsonl(intents):
    """
    Saves dataset in JSONL format.

    Each line represents one structured intent.

    JSONL is efficient for:
    - streaming training data
    - large datasets
    - incremental dataset updates

    Parameters
    ----------
    intents : list
    """

    output_path = INTENTS_DATA_DIR / "structural_intents.jsonl"

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_path, "w", encoding="utf-8") as f:

        for intent in intents:

            f.write(intent.model_dump_json())

            f.write("\n")

    print(f"Dataset saved to: {output_path}")


def print_section(title):

    print("\n" + "=" * 60)
    print(title.upper())
    print("=" * 60)


def print_subsection(title):

    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)


def print_kv(key, value):

    print(f"{key:<35}: {value}")


def main():
    """
    Orchestrates the full structural dataset generation pipeline.
    """

    print_section("Structural Dataset Generation")

    print_subsection("Configuration")

    config = load_generation_config()

    print_kv("semantic_balance", config.enable_semantic_balance)

    print_kv("min_per_bucket", getattr(config, "min_per_bucket", None))

    print_kv("max_per_bucket", getattr(config, "max_per_bucket", None))


    print_subsection("Schema")

    schema = load_schema()


    print_subsection("Intent Generation")

    intents = generate_intents_by_regime(
        schema,
        config
    )


    print_subsection("Validation")

    intents = filter_valid_intents(intents)


    print_subsection("Semantic Balance")

    intents = apply_semantic_balance_by_regime(
        intents,
        config,
        top_k_regimes=6
    )


    print_subsection("Export")

    export_jsonl(intents)


    print("\nDone.")


if __name__ == "__main__":

    main()
