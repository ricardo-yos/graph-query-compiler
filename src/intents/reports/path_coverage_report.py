"""
Path Coverage Report
====================

Computes structural coverage diagnostics for a dataset of graph query intents.

The goal of this report is to evaluate how well the dataset explores
the structural space defined by the graph schema.

Metrics include:

- frequency of unique graph paths
- distribution of relationships
- node coverage
- path depth distribution
- target node distribution
- structural entropy (diversity indicator)

These diagnostics help identify:

- structural imbalance
- over-represented query patterns
- insufficient path diversity
- biases in target node selection

Expected input format:
    list of intents, each containing a "schema_spec" field
    describing the graph query structure.
"""

from collections import Counter
import math


def path_coverage_report(intents):
    """
    Computes structural statistics describing graph path usage.

    Parameters
    ----------
    intents : list[dict]
        List of intent objects containing schema specifications.

    Returns
    -------
    dict
        Dictionary containing structural coverage metrics such as:
        - unique paths
        - node distribution
        - relationship usage
        - depth distribution
        - entropy measures
    """

    # Counters for structural statistics
    path_counter = Counter()
    relationship_counter = Counter()
    depth_counter = Counter()
    target_counter = Counter()
    node_counter = Counter()

    for intent in intents:

        schema = intent["schema_spec"]

        # path = ordered sequence of graph traversal steps
        path = schema.get("path", [])

        # --------------------------------------------------
        # Convert path to a hashable representation
        # Allows counting identical structural paths
        # --------------------------------------------------
        path_tuple = tuple(
            (
                step["source"]["label"],
                step["relationship"],
                step["target"]["label"],
            )
            for step in path
        )

        path_counter[path_tuple] += 1

        # --------------------------------------------------
        # Count node and relationship occurrences
        # Measures structural component usage frequency
        # --------------------------------------------------
        for step in path:

            relationship_counter[step["relationship"]] += 1

            node_counter[step["source"]["label"]] += 1
            node_counter[step["target"]["label"]] += 1

        # --------------------------------------------------
        # Path depth = number of edges in traversal
        # Indicates structural complexity
        # --------------------------------------------------
        depth_counter[len(path)] += 1

        # --------------------------------------------------
        # Final target node of the query
        # Helps detect bias in prediction targets
        # --------------------------------------------------
        target_label = schema["target"]["label"]

        target_counter[target_label] += 1
        node_counter[target_label] += 1

    total = len(intents)

    # --------------------------------------------------
    # Entropy = diversity metric
    # Higher entropy -> more balanced distribution
    # --------------------------------------------------
    def entropy(counter):
        """
        Computes Shannon entropy for a frequency distribution.
        """

        total_count = sum(counter.values())

        if total_count == 0:
            return 0.0

        return -sum(

            (c / total_count) * math.log2(c / total_count)

            for c in counter.values()
        )

    # relative frequency of nodes across dataset
    node_coverage = {

        node: round(count / total, 3)

        for node, count in node_counter.items()
    }

    # --------------------------------------------------
    # Return structured diagnostic metrics
    # --------------------------------------------------
    return {

        "total_intents":
            total,

        "unique_paths":
            len(path_counter),

        "top_paths":
            path_counter.most_common(5),

        # distribution of final query targets
        "start_nodes":
            dict(target_counter),

        "node_coverage":
            node_coverage,

        "relationship_usage":
            dict(relationship_counter),

        "depth_distribution":
            dict(depth_counter),

        "unique_targets":
            len(target_counter),

        "target_distribution":
            dict(target_counter),

        "target_rate":
            {
                k: round(v / total, 3)
                for k, v in target_counter.items()
            },

        # diversity indicators
        "path_entropy":
            round(entropy(path_counter), 4),

        "target_entropy":
            round(entropy(target_counter), 4),
    }
