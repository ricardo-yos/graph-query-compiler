"""
Environment Loader Utility
==========================

This module provides a utility function for loading environment variables
from a `.env` file into the system environment.

It is designed to be used as an explicit initialization step before
accessing any configuration that depends on environment variables
(e.g., database connections, API keys, or external services).

Dependencies
------------
- python-dotenv
- config.paths (ENV_PATH)

Example
-------
>>> from config.env_loader import load_env
>>> load_env()
"""

import os
from dotenv import load_dotenv

from config.paths import ENV_PATH


# ==================================================
# Environment loading
# ==================================================

def load_env() -> None:
    """
    Load environment variables from a `.env` file into the system environment.

    This function reads all variables from the `.env` file located at the
    path specified by `ENV_PATH` and injects them into `os.environ`.

    Raises
    ------
    FileNotFoundError
        If the `.env` file does not exist at `ENV_PATH`.
    """
    if not os.path.exists(ENV_PATH):
        raise FileNotFoundError(
            f".env file not found at expected path: {ENV_PATH}"
        )

    load_dotenv(ENV_PATH)
