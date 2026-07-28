"""
GGUF Conversion Pipeline
========================

Utility script responsible for converting a Hugging Face model
into GGUF format for llama.cpp inference.

Workflow
--------
1. Locate merged Hugging Face model
2. Execute llama.cpp conversion utility
3. Export model weights into GGUF format

Output
------
GGUF model file generated inside:

models/gguf/

Requirements
------------
Requires llama.cpp repository containing:

convert_hf_to_gguf.py
"""

from pathlib import Path
import subprocess


from config.paths import MODELS_DIR


# ============================================================
# Conversion configuration
# ============================================================

# Hugging Face model directory produced after QLoRA merging.
MERGED_MODEL_DIR = Path(MODELS_DIR) / "gqc-merged"


# Directory where the GGUF model will be exported.
GGUF_OUTPUT_PATH = Path(MODELS_DIR) / "gguf"


# Local llama.cpp repository containing conversion scripts.
LLAMA_CPP_DIR = Path("~/llama.cpp").expanduser()


# ============================================================
# Conversion utility
# ============================================================

def convert_to_gguf(
    model_path: str,
    output_dir: str,
    quantization: str = "f16",
) -> None:
    """
    Convert a Hugging Face model into GGUF format.

    Uses llama.cpp conversion utilities to transform a
    Transformers-compatible model into a format optimized
    for llama.cpp inference.

    Parameters
    ----------
    model_path : str
        Path to the Hugging Face model directory.

    output_dir : str
        Directory where the GGUF file will be generated.

    quantization : str, default="f16"
        Output precision format.

    Notes
    -----
    Requires the llama.cpp script:

    convert_hf_to_gguf.py

    available inside the llama.cpp repository.
    """

    model_path = Path(model_path)
    output_dir = Path(output_dir)


    # --------------------------------------------------------
    # Prepare output directory
    # --------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # --------------------------------------------------------
    # Build llama.cpp conversion command
    # --------------------------------------------------------

    command = [
        "python",
        str(
            LLAMA_CPP_DIR /
            "convert_hf_to_gguf.py"
        ),
        str(model_path),
        "--outfile",
        str(
            output_dir /
            f"{model_path.name}-{quantization}.gguf"
        ),
        "--outtype",
        quantization,
    ]


    # --------------------------------------------------------
    # Execute conversion
    # --------------------------------------------------------

    print("Exporting model to GGUF...")

    subprocess.run(
        command,
        check=True,
    )


    print(
        "GGUF export completed successfully!"
    )

    print(
        f"GGUF file saved to: {GGUF_OUTPUT_PATH.resolve()}"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    convert_to_gguf(
        model_path=MERGED_MODEL_DIR,
        output_dir=GGUF_OUTPUT_PATH,
    )
