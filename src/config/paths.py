"""
Project Paths Configuration
===========================

Centralized filesystem path definitions for the Graph Query Compiler.

This module provides a single source of truth for project directories,
avoiding hardcoded paths across the codebase.

Responsibilities:
- define root project location
- organize dataset, model, source, and test directories
- provide shared paths for configuration files
- simplify portability and future refactoring

All paths use pathlib.Path for consistent and cross-platform
filesystem handling.

These constants are consumed by dataset generation, benchmark
evaluation, model training, inference, and query compilation modules.
"""

from pathlib import Path


# =============================================================================
# Root directory
# =============================================================================

# Project root directory resolved from the current module location.
ROOT_DIR = Path(__file__).resolve().parents[2]


# =============================================================================
# Top-level directories
# =============================================================================

# Main project directories.
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
SRC_DIR = ROOT_DIR / "src"


# =============================================================================
# Data subdirectories
# =============================================================================

# Dataset storage, generated intents, schemas, and reports.
BENCHMARK_DATA_DIR = DATA_DIR / "benchmark"
DATASETS_DATA_DIR = DATA_DIR / "datasets"
INTENTS_DATA_DIR = DATA_DIR / "intents"
REPORTS_DIR = DATA_DIR / "reports"
SCHEMA_DATA_DIR = DATA_DIR / "schema"


# =============================================================================
# Benchmark structure
# =============================================================================

# Benchmark examples and evaluation reports.
BENCHMARK_EXAMPLES_DIR = BENCHMARK_DATA_DIR / "benchmark_breakdown"
BENCHMARK_REPORTS_DIR = BENCHMARK_DATA_DIR / "reports"


# =============================================================================
# Dataset structure
# =============================================================================

# Dataset organization by generation stage.
AUGMENTED_DATASETS_DIR = DATASETS_DATA_DIR / "augmented"
BASE_DATASETS_DIR = DATASETS_DATA_DIR / "base"
SPLITS_DATASETS_DIR = DATASETS_DATA_DIR / "splits"


# =============================================================================
# Model storage
# =============================================================================

# Directory containing the trained QLoRA adapter.
LORA_ADAPTER_DIR = MODELS_DIR / "qlora-intent-model"


# =============================================================================
# Configuration directories
# =============================================================================

# Configuration files grouped by project component.
CONFIG_DIR = SRC_DIR / "config"
DATASETS_CONFIG_DIR = CONFIG_DIR / "datasets"
INTENTS_CONFIG_DIR = CONFIG_DIR / "intents"
FINE_TUNING_CONFIG_DIR = CONFIG_DIR / "fine_tuning"
INFERENCE_CONFIG_DIR = CONFIG_DIR / "inference"


# =============================================================================
# Environment configuration
# =============================================================================

# Environment variables file.
ENV_PATH = ROOT_DIR / ".env"
