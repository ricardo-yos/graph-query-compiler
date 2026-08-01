"""
Aggregation policy defining valid aggregatable attributes
and supported aggregation functions.

Ensures semantic consistency for aggregate query generation.
"""

AGGREGATABLE_ATTRIBUTES = {

    "Place": [
        "rating",
        "num_reviews",
    ],

    "Road": [
        "length",
        "maxspeed",
    ],

    "Neighborhood": [
        "area_km2",
        "average_monthly_income",
        "literacy_rate",
        "population_with_income",
        "total_literate_population",
        "total_private_households",
        "total_resident_population",
        "street_count",
    ],
}

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
