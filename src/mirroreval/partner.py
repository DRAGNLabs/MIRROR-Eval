"""
Partner Model Backends
======================
Provides a unified interface for partner models used in multi-turn
conversation simulation. Two backends:

  - LocalPartner:  wraps a HuggingFace text-generation pipeline
  - OpenAIPartner: wraps the OpenAI chat completions API

Both expose __call__(messages, **kwargs) and return the same shape so
simulate_conversation doesn't need to know which backend it's talking to.
"""

from mirroreval.hf_utilities import get_hf_pipeline
from mirroreval.logger import logger


class LocalPartner:
    """Partner backed by a locally-loaded HuggingFace pipeline."""

    def __init__(self, model_name, quantize=None):
        logger.info(f"Loading local partner: {model_name} (quantize={quantize})")
        self.pipe = get_hf_pipeline(model_name, quantize=quantize)
        self.model_name = model_name

    def __call__(self, messages, **kwargs):
        output = self.pipe(messages, max_new_tokens=kwargs.get("max_new_tokens", 256), max_length=None)
        return output[0]["generated_text"][-1]["content"]


class OpenAIPartner:
    """Partner backed by the OpenAI chat completions API.

    Reads OPENAI_API_KEY from the environment automatically.
    """

    def __init__(self, model_name="gpt-4o-mini"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model_name = model_name
        logger.info(f"Using OpenAI partner: {model_name}")

    def __call__(self, messages, **kwargs):
        max_tokens = kwargs.get("max_new_tokens", 256)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content


def get_partner(backend, model_name, quantize=None):
    """Factory that returns the right partner based on config.

    Args:
        backend: "local" or "openai".
        model_name: HF model ID (local) or OpenAI model name (openai).
        quantize: "4bit", "8bit", or None. Only used for local backend.
    """
    if backend == "openai":
        return OpenAIPartner(model_name)
    elif backend == "local":
        return LocalPartner(model_name, quantize=quantize)
    else:
        raise ValueError(f"Unknown partner backend: {backend!r}. Use 'local' or 'openai'.")
