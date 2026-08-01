"""
Filter policy defining which attributes can be used in filter clauses.
"""

FILTERABLE_ATTRIBUTES = {

    "Place": {
        "rating",
        "num_reviews",
        "type",
    },

    "Neighborhood": {
        "name",
        "area_km2",
        "average_monthly_income",
        "total_resident_population",
    },

    "Road": {
        "name",
        "highway",
        "length",
        "maxspeed",
        "oneway",
    },

    "Intersection": {
        "highway",
        "street_count",
    },

    "Review": {
        "author",
        "rating",
        "date",
    },
}


"""
Filter value policy defining valid ranges and categorical values.
"""

FILTER_VALUE_RANGES = {

    "Neighborhood": {

        "average_monthly_income": {
            "type": "range",
            "min": 1000,
            "max": 12000
        },

        "area_km2": {
            "type": "range",
            "min": 1,
            "max": 20
        },

        "total_resident_population": {
            "type": "discrete",
            "values": [5000, 10000, 20000, 50000]
        },

        "name": {
            "type": "categorical",
            "values": ["Centro", "Jardim Paulista", "Vila Mariana", "Moema", "Pinheiros"]
        },
    },

    "Place": {

        "rating": {
            "type": "discrete",
            "values": [2, 3, 4, 5]
        },

        "num_reviews": {
            "type": "range",
            "min": 5,
            "max": 300
        },

        "name": {
            "type": "categorical",
            "values": ["Pet Shop Amigo Fiel", "Clínica Veterinária São Francisco", "Pet Care Center", "Mundo Pet", "Casa do Pet"]
        },

        "type": {
            "type": "categorical",
            "values": ["pet_store", "veterinary_care"]
        },
    },

    "Road": {

        "length": {
            "type": "range",
            "min": 50,
            "max": 2000
        },

        "maxspeed": {
            "type": "discrete",
            "values": [30, 40, 50, 60, 80]
        },

        "name": {
            "type": "categorical",
            "values": ["Rua das Flores", "Avenida Paulista", "Rua Augusta", "Rua da Consolação"]
        },
    },

    "General": {

        "date": {
            "type": "discrete",
            "values": [2020, 2021, 2022, 2023]
        },
    }
}


"""
Mandatory filters applied to specific node labels.
"""

MANDATORY_FILTERS = {

    "Place": [

        {
            "attribute": "type",
            "operator": "=",
        }

    ]

}


"""
Filter generation policy defining when optional filters
should be generated.
"""

FILTER_GENERATION_POLICY = {

    "simple_lookup_query": {
        "allow_optional_filters": True,
    },

    "simple_count_query": {
        "allow_optional_filters": True,
    },

    "simple_aggregation_query": {
        "allow_optional_filters": True,
    },

    "simple_ranking_query": {
        "allow_optional_filters": True,
    },


    "relational_lookup_query": {
        "allow_optional_filters": True,
    },

    "relational_count_query": {
        "allow_optional_filters": True,
    },

    "relational_aggregation_query": {
        "allow_optional_filters": True,
    },

    "relational_ranking_query": {
        "allow_optional_filters": True,
    },
}
