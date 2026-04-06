"""
Operator policy defining valid operators per attribute.
"""

NODE_ATTRIBUTE_OPERATORS = {

    "Neighborhood": {

        "name": ["=", "contains"],

        "literacy_rate": [">", "<", ">=", "<="],
        "average_monthly_income": [">", "<", ">=", "<="],
        "area_km2": [">", "<", ">=", "<="],

        "population_with_income": ["=", ">", "<"],
        "total_resident_population": ["=", ">", "<"],
        "total_literate_population": ["=", ">", "<"],
    },

    "Place": {

        "name": ["=", "contains"],
        "type": ["="],

        "rating": [">", "<", ">=", "<="],
        "num_reviews": [">", "<", ">=", "<="],
    },

    "Road": {

        "name": ["=", "contains"],

        "length": [">", "<", ">=", "<="],
        "maxspeed": [">", "<", ">=", "<="],
    },

    "General": {

        "date": [">", "<", ">=", "<="],
    }
}