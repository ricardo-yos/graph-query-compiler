"""
Project Directory Constants
===========================

Defines and centralizes all filesystem paths used throughout the
Graph Query Compiler project.

This module provides a single source of truth for directory and file
locations, ensuring:

- Consistent path management across modules
- Reduced hardcoded paths
- Easier refactoring and portability
- Clear project structure documentation

All paths are implemented using `pathlib.Path` to guarantee
cross-platform compatibility and improved readability.

The constants defined here are used by data processing pipelines,
model training, semantic compilation, configuration loading,
and environment setup.
"""

from pathlib import Path


# =============================================================================
# Root Directory
# =============================================================================

# Absolute path to the project root directory.
# It is computed relative to this file to ensure portability.
ROOT_DIR = Path(__file__).resolve().parents[2]


# =============================================================================
# Top-Level Project Directories
# =============================================================================

DATA_DIR = ROOT_DIR / "data"        # All project data (raw, processed, schema, etc.)
MODELS_DIR = ROOT_DIR / "models"    # Trained models and adapters
SRC_DIR = ROOT_DIR / "src"          # Source code root
TESTS_DIR = ROOT_DIR / "tests"      # Test suite directory


# =============================================================================
# Data Subdirectories
# =============================================================================

DATASETS_DATA_DIR = DATA_DIR / "datasets"   # Datasets for training and evaluation
INTENTS_DATA_DIR = DATA_DIR / "intents"     # Intent classification datasets
SCHEMA_DATA_DIR = DATA_DIR / "schema"       # Graph schema definitions


# =============================================================================
# Intent Dataset Subdirectories
# =============================================================================

CLEANED_INTENTS_DIR = INTENTS_DATA_DIR / "cleaned"        # Cleaned intent data
RAW_INTENTS_DIR = INTENTS_DATA_DIR / "raw"                # Raw collected intents
VALIDATED_INTENTS_DIR = INTENTS_DATA_DIR / "validated"    # Validated and curated intents


# =============================================================================
# Main Dataset Subdirectories
# =============================================================================

AUGMENTED_DATASETS_DIR = DATASETS_DATA_DIR / "augmented"  # Data augmentation outputs
BASE_DATASETS_DIR = DATASETS_DATA_DIR / "base"            # Original base datasets
SPLITS_DATASETS_DIR = DATASETS_DATA_DIR / "splits"        # Train/validation/test splits


# =============================================================================
# Model Storage
# =============================================================================

LORA_ADAPTER_DIR = (
    MODELS_DIR / "qlora-intent-model"
)  # Fine-tuned LoRA adapter directory


# =============================================================================
# Configuration Directories
# =============================================================================

CONFIG_DIR = SRC_DIR / "config"                      # General configuration folder
DATASETS_CONFIG_DIR = CONFIG_DIR / "datasets"        # Dataset configuration files
FINE_TUNING_CONFIG_DIR = CONFIG_DIR / "fine_tuning"  # Fine-tuning configuration files


# =============================================================================
# Environment File
# =============================================================================

ENV_PATH = ROOT_DIR / ".env"  # Environment variables file
