"""Recipe Step Autocompleter Package.

Helps home cooks autocomplete incomplete recipe instructions using
statistical n-gram language models with Maximum-Likelihood Estimation
and advanced smoothing techniques (Laplace, Lidstone, Backoff).
"""

from app.core import (
    END_TOKEN,
    START_TOKEN,
    UNK_TOKEN,
    NGramModel,
    detokenize,
    pad_sequence,
    preprocess_text,
    tokenize,
)
from app.llm import generate_recipe_step_llm, is_llm_available, llm_hello
from app.storage import (
    append_recipe_step,
    load_model_artifact,
    load_recipe_corpus,
    save_model_artifact,
)

__version__ = "1.0.0"
__all__ = [
    "NGramModel",
    "preprocess_text",
    "tokenize",
    "detokenize",
    "pad_sequence",
    "START_TOKEN",
    "END_TOKEN",
    "UNK_TOKEN",
    "load_recipe_corpus",
    "append_recipe_step",
    "save_model_artifact",
    "load_model_artifact",
    "generate_recipe_step_llm",
    "is_llm_available",
    "llm_hello",
]
