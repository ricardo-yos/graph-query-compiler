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
        "max_paths": 1800,
        "max_projection_samples": 20,
        "max_filter_combinations": 100,
        "max_order_attributes": 0,
        "max_limit_variants": 0,
    },

    "relational_count_query": {
        "max_paths": 1600,
        "max_projection_samples": 12,
        "max_filter_combinations": 100,
        "max_order_attributes": 0,
        "max_limit_variants": 0,
    },

    "relational_aggregation_query": {
        "max_paths": 2200,
        "max_projection_samples": 20,
        "max_filter_combinations": 120,
        "max_order_attributes": 1,
        "max_limit_variants": 0,
    },

    "relational_ranking_query": {
        "max_paths": 350,
        "max_projection_samples": 4,
        "max_filter_combinations": 10,
        "max_order_attributes": 1,
        "max_limit_variants": 1,
    },
}
