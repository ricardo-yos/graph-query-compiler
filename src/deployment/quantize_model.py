"""
GGUF Quantization Pipeline
==========================

Utility script responsible for quantizing GGUF models using
llama.cpp quantization tools.

Workflow
--------
1. Load FP16 GGUF model
2. Execute llama-quantize utility
3. Generate optimized quantized GGUF model

Output
------
Quantized GGUF model generated at:

models/gguf/

Requirements
------------
Requires llama.cpp compiled with:

llama-quantize
"""


import subprocess
from pathlib import Path


from config.paths import MODELS_DIR


# ============================================================
# Quantization configuration
# ============================================================

# Input FP16 GGUF model generated during export step.
GGUF_INPUT_PATH = (
    Path(MODELS_DIR)
    / "gguf"
    / "gqc-merged-f16.gguf"
)


# Local llama.cpp binary used for quantization.
LLAMA_CPP_DIR = Path("~/llama.cpp").expanduser()


QUANTIZE_BIN = (
    LLAMA_CPP_DIR
    / "build"
    / "bin"
    / "llama-quantize"
)


# Output quantized GGUF model.
GGUF_OUTPUT_PATH = (
    Path(MODELS_DIR)
    / "gguf"
    / "gqc-q4_k_m.gguf"
)


# ============================================================
# Quantization utility
# ============================================================

def quantize_model(
    input_model: str,
    output_model: str,
    quantization: str = "Q4_K_M",
) -> None:
    """
    Quantize a GGUF model using llama.cpp.

    Converts a higher precision GGUF model into a smaller
    quantized representation optimized for inference.

    Parameters
    ----------
    input_model : str
        Path to the input GGUF model.

    output_model : str
        Path where the quantized GGUF model will be saved.

    quantization : str, default="Q4_K_M"
        Quantization format used by llama-quantize.

    Notes
    -----
    Requires llama.cpp compiled with the llama-quantize binary.
    """

    input_model = Path(input_model)
    output_model = Path(output_model)


    # --------------------------------------------------------
    # Validate input model
    # --------------------------------------------------------

    if not input_model.exists():
        raise FileNotFoundError(
            f"Input model not found: {input_model}"
        )


    if not QUANTIZE_BIN.exists():
        raise FileNotFoundError(
            f"Quantization binary not found: {QUANTIZE_BIN}"
        )


    # --------------------------------------------------------
    # Build quantization command
    # --------------------------------------------------------

    command = [
        str(QUANTIZE_BIN),
        str(input_model),
        str(output_model),
        quantization,
    ]


    # --------------------------------------------------------
    # Execute quantization
    # --------------------------------------------------------

    print(
        f"Quantizing model using {quantization}..."
    )

    subprocess.run(
        command,
        check=True,
    )


    print(
        "Quantization completed successfully!"
    )

    print(
        f"Quantized model saved to: "
        f"{output_model.resolve()}"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    quantize_model(
        input_model=GGUF_INPUT_PATH,
        output_model=GGUF_OUTPUT_PATH,
        quantization="Q4_K_M",
    )
