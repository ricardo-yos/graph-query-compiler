"""
SchemaNormalizer Exhaustive Test Suite
=======================================

Comprehensive test module validating structural normalization behavior
of the SchemaNormalizer component within the graph query compiler pipeline.

Purpose
-------
This test suite ensures that the SchemaNormalizer:

1. Enforces structural consistency of the intermediate representation (IR).
2. Guarantees presence of all mandatory schema fields.
3. Normalizes malformed or partially defined LLM-generated schemas.
4. Applies correct default types and values.
5. Prevents downstream runtime failures caused by schema shape inconsistencies.

Testing Strategy
----------------
The suite performs exhaustive combinatorial testing by generating
all possible combinations of:

- constraints
- known
- path
- return

Each field is tested against multiple malformed, incomplete,
or edge-case configurations.

For every generated schema, the test asserts:

- user_intent defaults to "retrieve".
- schema structure always conforms to DEFAULT_SCHEMA.
- constraints subfields are normalized to lists.
- known is always a list of dictionaries.
- path is always a list.
- return is always a dictionary.

Why This Matters
----------------
LLMs frequently generate incomplete or structurally inconsistent schemas.
This test suite guarantees that the normalization layer acts as a
deterministic structural firewall before semantic validation stages.

Scope
-----
This module validates structural normalization only.
It does NOT test semantic resolution or grounding behavior.
"""

import pytest
from copy import deepcopy
from src.compiler.normalization.normalizer import SchemaNormalizer  # adjust to your project structure
from itertools import product

# --- Keys in the schema that will be tested ---
SCHEMA_KEYS = ["constraints", "known", "path", "return"]
CONSTRAINT_KEYS = ["filters", "limit", "order_by"]

# --- Test values for each schema field ---
FIELD_TEST_VALUES = {
    "constraints": [None, {}, {"filters": {"field": "name"}}, {"filters": [], "limit": None}],
    "known": [None, {}, [{"id": 1}], [{"id": 2}, {"id": 3}]],
    "path": [None, [], ["node1", "node2"]],
    "return": [None, {}, {"field": "value"}],
}

def generate_schema_combinations():
    """
    Generate all combinations of partial schema fields for testing.

    Returns
    -------
    List[dict]
        List of schema dictionaries with all combinations of test values.
    """
    combos = list(product(*FIELD_TEST_VALUES.values()))
    schema_combos = []
    for combo in combos:
        schema = dict(zip(SCHEMA_KEYS, combo))
        schema_combos.append(schema)
    return schema_combos

@pytest.mark.parametrize("partial_schema", generate_schema_combinations())
def test_exhaustive_normalization(partial_schema):
    """
    Test SchemaNormalizer.normalize with exhaustive combinations of partial schemas.

    Parameters
    ----------
    partial_schema : dict
        A partial or malformed schema dictionary from the generated combinations.

    Checks
    ------
    - The normalized schema is always a dict.
    - 'user_intent' defaults to 'retrieve'.
    - All default keys exist in the schema.
    - Nested fields (constraints, known, path, return) have correct types:
        - constraints: dict with lists for subfields
        - known: list of dicts
        - path: list
        - return: dict
    """
    # Prepare input for normalizer
    ir_input = {"schema": deepcopy(partial_schema)}
    
    # Normalize
    normalized = SchemaNormalizer.normalize(deepcopy(ir_input))
    
    schema = normalized["schema"]
    
    # Check default user intent
    assert normalized["user_intent"] == "retrieve"
    
    # Schema must always be a dict
    assert isinstance(schema, dict)
    
    # Check that all default keys exist
    for key in SchemaNormalizer.DEFAULT_SCHEMA.keys():
        assert key in schema
    
    # Check constraints structure
    constraints = schema["constraints"]
    assert isinstance(constraints, dict)
    for key in CONSTRAINT_KEYS:
        val = constraints.get(key)
        assert isinstance(val, list)
    
    # Check known field
    known = schema["known"]
    assert isinstance(known, list)
    for k in known:
        assert isinstance(k, dict)
    
    # Check path field
    path = schema["path"]
    assert isinstance(path, list)
    
    # Check return field
    ret = schema["return"]
    assert isinstance(ret, dict)

# --- Direct execution with print output ---
if __name__ == "__main__":
    import sys
    retcode = pytest.main([__file__, "-v"])
    if retcode == 0:
        print("\nAll exhaustive SchemaNormalizer tests passed successfully!")
    else:
        print("\nSome tests failed. Check above for details.")
        sys.exit(retcode)
