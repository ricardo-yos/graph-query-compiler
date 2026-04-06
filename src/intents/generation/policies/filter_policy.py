"""
Filter policy defining which attributes can be used in filter clauses.
"""

FILTERABLE_ATTRIBUTES = {

    "Place": {
        "name",
        "rating",
        "num_reviews",
        "type",
        "latitude",
        "longitude",
    },

    "Neighborhood": {
        "name",
        "area_km2",
        "average_monthly_income",
        "literacy_rate",
        "population_with_income",
        "total_literate_population",
        "total_private_households",
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

        "literacy_rate": {
            "type": "range",
            "min": 50,
            "max": 100
        },

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

        "population_with_income": {
            "type": "discrete",
            "values": [1000, 3000, 5000, 10000, 50000]
        },

        "total_resident_population": {
            "type": "discrete",
            "values": [5000, 10000, 20000, 50000]
        },

        "total_literate_population": {
            "type": "discrete",
            "values": [4000, 8000, 15000, 20000]
        },

        "name": {
            "type": "categorical",
            "values": ["Centro", "Pinheiros", "Moema"]
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
            "values": ["Pet Shop Centro", "Clínica Vet", "Banho e Tosa"]
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
            "values": ["Rua A", "Avenida B"]
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