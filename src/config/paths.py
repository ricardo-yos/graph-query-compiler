"""
Project Paths Configuration
===========================

Centralizes all filesystem paths used across the Graph Query Compiler project.

This module acts as a single source of truth for directory and file locations,
providing:

- Consistent path management across all modules
- Elimination of hardcoded paths
- Easier refactoring and portability
- Clear and explicit project structure

All paths are defined using `pathlib.Path` to ensure cross-platform
compatibility and improved readability.

These constants are used by dataset pipelines, model training,
inference, configuration loading, and the query compiler.
"""

from pathlib import Path


# =============================================================================
# Root directory
# =============================================================================

# Absolute path to the project root directory.
# Computed relative to this file to ensure portability.
ROOT_DIR = Path(__file__).resolve().parents[2]


# =============================================================================
# Top-level directories
# =============================================================================

DATA_DIR = ROOT_DIR / "data"        # Stores datasets, schema, and reports
MODELS_DIR = ROOT_DIR / "models"    # Stores trained models and adapters
SRC_DIR = ROOT_DIR / "src"          # Source code root
TESTS_DIR = ROOT_DIR / "tests"      # Test suite


# =============================================================================
# Data subdirectories
# =============================================================================

DATASETS_DATA_DIR = DATA_DIR / "datasets"   # Training and evaluation datasets
INTENTS_DATA_DIR = DATA_DIR / "intents"     # Generated intent datasets
REPORTS_DIR = DATA_DIR / "reports"          # Generated reports and analytics
SCHEMA_DATA_DIR = DATA_DIR / "schema"       # Graph schema definitions


# =============================================================================
# Dataset structure
# =============================================================================

BASE_DATASETS_DIR = DATASETS_DATA_DIR / "base"     # Raw/base datasets
SPLITS_DATASETS_DIR = DATASETS_DATA_DIR / "splits" # Train/validation/test splits


# =============================================================================
# Model storage
# =============================================================================

# Directory for the fine-tuned LoRA adapter
LORA_ADAPTER_DIR = MODELS_DIR / "qlora-intent-model"


# =============================================================================
# Configuration directories
# =============================================================================

CONFIG_DIR = SRC_DIR / "config"                      # Root config directory
DATASETS_CONFIG_DIR = CONFIG_DIR / "datasets"        # Dataset configs
INTENTS_CONFIG_DIR = CONFIG_DIR / "intents"          # Intent configs
FINE_TUNING_CONFIG_DIR = CONFIG_DIR / "fine_tuning"  # Training/inference configs


# =============================================================================
# Environment configuration
# =============================================================================

ENV_PATH = ROOT_DIR / ".env"  # Environment variables file
