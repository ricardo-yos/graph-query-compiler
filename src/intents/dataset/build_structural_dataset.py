"""
Combinatorial Intent Generation Pipeline
========================================

Builds a structurally diverse dataset of graph query intents by exploring
combinations of schema-defined paths, projections, filters, and operators.

Generation strategy:

- explore traversal paths with different depths
- combine structural operators
- validate generated structures
- enforce semantic diversity balancing

Pipeline steps:

1. Load generation configuration
2. Load graph schema
3. Generate intents by structural regime
4. Validate generated structures
5. Apply semantic balancing
6. Export dataset as JSONL

Inputs:

- combinatorial.yaml
    generation policies and expansion limits

- regime_types.yaml
    structural regime definitions

- graph_schema.json
    graph entities, attributes, and relationships

Output:

- structural_intents.jsonl
    structured intents for NL → query training

Purpose:

Provide broad structural coverage for training models that map
natural language questions into graph query representations.
"""

import json
import random
from collections import Counter, defaultdict
import yaml

from config.paths import SCHEMA_DATA_DIR, INTENTS_DATA_DIR, INTENTS_CONFIG_DIR

from src.intents.generation.structural_config import StructuralGenerationConfig
from src.intents.generation.graph_schema_adapter import GraphSchemaAdapter
from src.intents.generation.combinatorial_generator import CombinatorialStructuralGenerator

from src.intents.validation.intent_validator import get_validation_result
from src.intents.dataset_curation.semantic_bucket_selector import SemanticBucketSelector


# Ensure deterministic dataset generation.
SEED = 42
random.seed(SEED)


def load_generation_config() -> StructuralGenerationConfig:
    """
    Load the global structural generation configuration.

    The configuration controls:

    - diversity balancing
    - projection and filter limits
    - combinatorial expansion constraints

    Returns
    -------
    StructuralGenerationConfig
        Validated configuration used by the generator.
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
    Load structural regimes defining query complexity patterns.

    Each regime controls which structural operators and constraints
    are available during intent generation.

    Returns
    -------
    dict
        Mapping between regime names and generation constraints.
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
    Load graph schema and create the internal schema adapter.

    The adapter provides access to:

    - entities
    - attributes
    - relationships
    - traversal rules

    Returns
    -------
    GraphSchemaAdapter
        Schema interface consumed by the intent generator.
    """

    schema_path = SCHEMA_DATA_DIR / "graph_schema.json"

    with open(schema_path, encoding="utf-8") as f:
        raw_schema = json.load(f)

    schema = GraphSchemaAdapter(raw_schema)

    print(f"Schema loaded from:\n{schema_path}")

    return schema


def build_config_from_regime(base_config, regime_name, regime):
    """
    Create a regime-specific generation configuration.

    Each regime defines a different structural complexity level,
    controlling available operators and expansion limits.

    Parameters
    ----------
    base_config : StructuralGenerationConfig
        Base generation configuration.

    regime_name : str
        Identifier of the structural regime.

    regime : dict
        Regime-specific constraints.

    Returns
    -------
    StructuralGenerationConfig
        Configuration customized for the selected regime.
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
    Generate intents across all configured structural regimes.

    Each regime is generated independently to ensure coverage
    of different query complexity patterns.

    Parameters
    ----------
    schema : GraphSchemaAdapter
        Graph schema used during intent generation.

    base_config : StructuralGenerationConfig
        Base configuration used to create regime-specific settings.

    Returns
    -------
    list
        Generated structural intent objects.
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
            config=regime_config,
        )

        intents = generator.generate()

        print(f"Generated: {len(intents)}")

        all_intents.extend(intents)

    print("\nTotal generated:", len(all_intents))

    return all_intents


def filter_valid_intents(intents):
    """
    Filter invalid generated intents using structural validation rules.

    Validation failures are collected by reason and regime to help
    identify generation bottlenecks.
    """

    valid_intents = []

    removed = 0

    rejection_counter = Counter()

    rejection_by_regime = defaultdict(Counter)

    for intent in intents:

        intent_dict = intent.model_dump()

        valid, reason = get_validation_result(intent_dict)

        if not valid and reason == "regime_path":
            print("=" * 80)
            print(intent.model_dump())

        if valid:

            valid_intents.append(intent)

        else:

            removed += 1

            rejection_counter[reason] += 1

            regime = intent.intent.regime

            rejection_by_regime[regime][reason] += 1

    print(f"Valid intents   : {len(valid_intents)}")

    print(f"Rejected intents: {removed}")

    print("\nOverall rejection reasons:")

    for reason, count in rejection_counter.most_common():

        print(f"  {reason:<25} {count}")

    print("\nRejections by regime:")

    for regime, counter in rejection_by_regime.items():

        print(f"\n{regime}")

        for reason, count in counter.most_common():

            print(f"  {reason:<25} {count}")

    return valid_intents


def apply_semantic_balance_by_regime(intents, config, top_k_regimes=3):
    """
    Apply semantic diversity balancing to high-volume regimes.

    The strategy reduces repetitive structural patterns by applying
    bucket-based sampling only where generation volume is highest.

    Parameters
    ----------
    intents : list
        Generated structural intents.

    config : StructuralGenerationConfig
        Generation configuration containing balancing rules.

    top_k_regimes : int
        Number of regimes selected for balancing.

    Returns
    -------
    list
        Semantically balanced intents.
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

    # Select regimes with highest generation volume.
    regimes_to_balance = {

        name for name, _ in regimes_sorted[:top_k_regimes]

    }

    print("Semantic balance applied to regimes:")

    for r in regimes_to_balance:

        print(" -", r)

    # Apply semantic bucket selection to reduce structural repetition.
    selector = SemanticBucketSelector(config)

    final_intents = []

    for regime, group in grouped.items():

        # Balance only selected high-volume regimes.
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
    Export generated intents to JSONL format.

    Each line contains one serialized structural intent,
    enabling efficient streaming and incremental dataset usage.

    Parameters
    ----------
    intents : list
        Intent objects to export.
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
    Execute the complete structural intent generation pipeline.

    Pipeline stages:

    - load configuration
    - load graph schema
    - generate intents
    - validate structures
    - balance semantic diversity
    - export dataset
    """

    print_section("Structural Dataset Generation")

    print_subsection("Configuration")

    # Load generation policies and expansion constraints.
    config = load_generation_config()

    print_kv("semantic_balance", config.enable_semantic_balance)

    print_kv("min_per_bucket", getattr(config, "min_per_bucket", None))

    print_kv("max_per_bucket", getattr(config, "max_per_bucket", None))


    print_subsection("Schema")

    # Load graph structure used for intent generation.
    schema = load_schema()


    print_subsection("Intent Generation")

    # Generate structural intents across all regimes.
    intents = generate_intents_by_regime(
        schema,
        config
    )


    print_subsection("Validation")

    # Remove structurally invalid intents.
    intents = filter_valid_intents(intents)

    print_subsection("Semantic Balance")

    # Reduce repetitive patterns in high-volume regimes.
    intents = apply_semantic_balance_by_regime(
        intents,
        config,
        top_k_regimes=6
    )

    print_subsection("Export")

    # Export final dataset for training usage.
    export_jsonl(intents)

    print("\nDone.") 


if __name__ == "__main__":

    main()
