"""
GGUF / llama.cpp inference pipeline for GQC.

This module loads a quantized GGUF model using llama.cpp,
generates structured predictions from natural language
questions, and returns the generated GQC intent representation.

Workflow
--------
1. Load inference configuration
2. Initialize llama.cpp model
3. Build prompt from user question
4. Generate model output
5. Extract JSON response
6. Return structured prediction containing:
   - regime
   - schema
"""


import json
import yaml

from pathlib import Path
from typing import Dict, Union

from llama_cpp import Llama

from config.paths import INFERENCE_CONFIG_DIR


# ==================================================
# Configuration loading
# ==================================================

# Load inference parameters from YAML configuration.
CONFIG_PATH = (
    Path(INFERENCE_CONFIG_DIR)
    / "llamacpp_config.yaml"
)

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)


# Model and generation parameters.
MODEL_PATH = cfg["model_path"]

MAX_NEW_TOKENS = cfg["max_new_tokens"]

STOP_TOKEN = cfg["stop_token"]

ANSWER_MARKER = cfg["answer_marker"]

TASK_INSTRUCTION = cfg["task_instruction"]


# Optional llama.cpp runtime parameters.
N_CTX = cfg.get("n_ctx", 4096)
N_THREADS = cfg.get("n_threads", 8)
N_GPU_LAYERS = cfg.get("n_gpu_layers", -1)



# ==================================================
# Model initialization
# ==================================================

# Initialize llama.cpp model once at module loading time.
# The same instance is reused during inference calls.
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=N_CTX,
    n_threads=N_THREADS,
    n_gpu_layers=N_GPU_LAYERS,
    verbose=False,
)



# ==================================================
# Prompt construction
# ==================================================

def build_prompt(question: str) -> str:
    """
    Build the instruction prompt used for inference.

    The prompt follows the training format to preserve
    consistency between fine-tuning and inference.

    Parameters
    ----------
    question : str
        Natural language query provided by the user.

    Returns
    -------
    str
        Formatted prompt for llama.cpp generation.
    """

    return (
        "### Instruction:\n"
        f"{TASK_INSTRUCTION}\n\n"
        "### Question:\n"
        f"{question}\n\n"
        "### Answer:\n"
    )



# ==================================================
# JSON extraction
# ==================================================

def extract_first_json(text: str) -> str:
    """
    Extract the first complete JSON object from generated text.

    LLM outputs may contain additional text before or after
    the JSON response. This function identifies the first
    balanced JSON object using bracket matching.

    Parameters
    ----------
    text : str
        Raw model output.

    Returns
    -------
    str
        Extracted JSON string.

    Raises
    ------
    ValueError
        If no valid JSON object is found.
    """

    start = text.find("{")

    if start == -1:
        raise ValueError(
            "No JSON object found in output"
        )

    stack = []

    for i in range(start, len(text)):

        if text[i] == "{":
            stack.append("{")

        elif text[i] == "}":

            stack.pop()

            if not stack:
                return text[start:i + 1]

    raise ValueError(
        "Unclosed JSON object in output"
    )



# ==================================================
# Model prediction
# ==================================================

def predict(
    question: str,
    debug: bool = False,
) -> Union[Dict, str]:
    """
    Generate a structured GQC prediction.

    The model output is parsed into a structured dictionary
    containing the predicted regime and schema.

    Parameters
    ----------
    question : str
        Natural language query.

    debug : bool, default=False
        If enabled, returns raw generation information
        instead of raising parsing exceptions.

    Returns
    -------
    dict
        Structured prediction containing:
        - regime
        - schema

    Raises
    ------
    Exception
        If the generated output cannot be parsed as JSON.
    """

    prompt = build_prompt(question)


    # Generate model response using deterministic decoding.
    output = llm(
        prompt,
        max_tokens=MAX_NEW_TOKENS,
        temperature=0.0,
        stop=[STOP_TOKEN],
        echo=False,
    )


    text = output["choices"][0]["text"]

    text = text.strip()


    try:

        # Extract and parse the generated JSON response.
        json_text = extract_first_json(text)

        parsed = json.loads(json_text)


        return {
            "regime": parsed.get("regime"),
            "schema": parsed.get("schema"),
        }


    except Exception as e:

        if debug:

            return {
                "error": "Invalid JSON generated",
                "raw_output": text,
                "exception": str(e),
            }

        raise



# ==================================================
# Command-line interface
# ==================================================

def main() -> None:
    """
    Run interactive inference from the command line.

    Allows manual testing of the GQC inference pipeline.
    """

    while True:

        question = input(
            "\nQuestion (ENTER to exit): "
        ).strip()


        if not question:
            break


        result = predict(
            question,
            debug=True,
        )


        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
