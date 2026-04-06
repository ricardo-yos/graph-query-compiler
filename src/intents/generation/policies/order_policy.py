"""
Order policy defining sortable attributes per node.
"""

ORDERABLE_ATTRIBUTES = {

    "Place": {
        "name",
        "rating",
        "num_reviews",
    },

    "Neighborhood": {
        "name",
        "area_km2",
        "average_monthly_income",
        "literacy_rate",
        "population_with_income",
        "total_resident_population",
    },

    "Road": {
        "name",
        "length",
        "maxspeed",
    },

    "Intersection": {
        "street_count",
    },

    "Review": {
        "rating",
        "date",
    },
}