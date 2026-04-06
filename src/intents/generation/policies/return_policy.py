"""
Return policy defining allowed output attributes.
"""

RETURN_POLICY = {

    "Place": {
        "primary": ["name"],
        "secondary": [
            ["name", "rating"],
            ["name", "num_reviews"]
        ],
        "allow_multi": True,
    },

    "Review": {
        "primary": ["text"],
        "secondary": [
            ["text", "rating"]
        ],
        "allow_multi": True,
    },

    "Neighborhood": {
        "primary": ["name"],
        "secondary": [],
        "allow_multi": False,
    },

    "Road": {
        "primary": ["name"],
        "secondary": [],
        "allow_multi": False,
    },

    "Intersection": {
        "primary": ["name"],
        "secondary": [],
        "allow_multi": False,
    },
}