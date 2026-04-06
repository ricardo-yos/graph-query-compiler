"""
Aggregate policy defining valid aggregation functions per attribute.
Ensures semantic consistency for aggregate queries.
"""

AGGREGATE_FUNCTIONS = {

    "rating": ["count", "avg"],

    "num_reviews": ["count", "sum"],

    "length": ["count", "avg", "sum"],

    "maxspeed": ["count", "avg"],

    "area_km2": ["count", "avg", "sum"],

    "average_monthly_income": ["count", "avg"],

    "literacy_rate": ["count", "avg"],

    "population_with_income": ["count", "sum"],

    "total_literate_population": ["count", "sum"],

    "total_private_households": ["count", "sum"],

    "total_resident_population": ["count", "sum"],

    "street_count": ["count", "sum"],
}