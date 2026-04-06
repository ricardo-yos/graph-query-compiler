"""
Path policy defining allowed traversal structure.
"""

PATH_POLICY = {

    "Neighborhood": {

        "allowed_targets": {"Place", "Road"},
        "max_depth": 2,
        "allow_cycles": False,
    },

    "Place": {

        "allowed_targets": {"Review", "Intersection", "Neighborhood"},
        "max_depth": 3,
        "allow_cycles": False,
    },

    "Road": {

        "allowed_targets": {"Place", "Intersection"},
        "max_depth": 3,
        "allow_cycles": False,
    },

    "Intersection": {

        "allowed_targets": {"Road"},
        "max_depth": 3,
        "allow_cycles": False,
    },

    "Review": {

        "allowed_targets": {"Place"},
        "max_depth": 2,
        "allow_cycles": False,
    },
}