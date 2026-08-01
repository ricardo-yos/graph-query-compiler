"""
Regime-specific structural expansion policies controlling
combinatorial sampling and query generation limits.
"""

REGIME_POLICY = {

    "simple_lookup_query": {
        "max_paths": 2000,
        "max_projection_samples": 25,
        "max_filter_combinations": 120,
        "max_order_attributes": 0,
        "max_limit_variants": 0,
    },

    "simple_count_query": {
        "max_paths": 1800,
        "max_projection_samples": 12,
        "max_filter_combinations": 140,
        "max_order_attributes": 0,
        "max_limit_variants": 0,
    },

    "simple_aggregation_query": {
        "max_paths": 2500,
        "max_projection_samples": 30,
        "max_filter_combinations": 180,
        "max_order_attributes": 1,
        "max_limit_variants": 0,
    },

    "simple_ranking_query": {
        "max_paths": 800,
        "max_projection_samples": 6,
        "max_filter_combinations": 20,
        "max_order_attributes": 1,
        "max_limit_variants": 1,
    },

    "relational_lookup_query": {
        "max_paths": 5000,
        "max_projection_samples": 60,
        "max_filter_combinations": 500,
        "max_order_attributes": 0,
        "max_limit_variants": 0,
    },

    "relational_count_query": {
        "max_paths": 4500,
        "max_projection_samples": 40,
        "max_filter_combinations": 500,
        "max_order_attributes": 0,
        "max_limit_variants": 0,
    },

    "relational_aggregation_query": {
        "max_paths": 5000,
        "max_projection_samples": 60,
        "max_filter_combinations": 600,
        "max_order_attributes": 2,
        "max_limit_variants": 0,
    },

    "relational_ranking_query": {
        "max_paths": 1200,
        "max_projection_samples": 12,
        "max_filter_combinations": 80,
        "max_order_attributes": 1,
        "max_limit_variants": 1,
    },
}
