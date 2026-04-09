"""
Creativity Embedding Utilities
==============================
Helpers for loading embedding models and converting text rows into vectors.

The creativity benchmark uses transformer embeddings rather than a generative
judge model. These utilities keep the embedding logic self-contained within
the benchmark package while following the same configuration-driven workflow
used elsewhere in MIRROR-Eval.
"""

from __future__ import annotations

from typing import Literal, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


EMBEDDING_MODEL_PRESETS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "bert-large-uncased": "bert-large-uncased",
    "bert-large": "bert-large-uncased",
}


def resolve_embedding_model_name(model_name: str) -> str:
    """
    Resolve shorthand config values to full Hugging Face model identifiers.

    This allows settings.toml to use convenient aliases such as `minilm`
    without hard-coding those names throughout the benchmark.
    """
    key = model_name.strip().lower()
    return EMBEDDING_MODEL_PRESETS.get(key, model_name.strip())


def mean_pooling(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool token embeddings into one vector per input sequence."""
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class TextEmbedder:
    """
    Embed text strings with a transformer encoder and mean pooling.

    The class loads the tokenizer/model pair once and then reuses them across
    the entire benchmark run, which is much more efficient than recreating the
    model for each dataset or metric call.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Initialize the embedding model and tokenizer.

        Args:
            model_name: Embedding model id or shorthand preset.
            device: Optional device override. Defaults to CUDA if available,
                otherwise CPU.
        """
        resolved_model_name = resolve_embedding_model_name(model_name)
        self.model_name = resolved_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(resolved_model_name)
        self.model = AutoModel.from_pretrained(resolved_model_name)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def embed(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
        max_length: int = 256,
        normalize: bool = True,
        show_progress: bool = False,
        progress_desc: str = "Embedding",
        return_type: Literal["numpy", "torch"] = "numpy",
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Compute embeddings for a sequence of texts.

        Args:
            texts: Ordered text rows to embed.
            batch_size: Number of texts per forward pass.
            max_length: Token truncation length.
            normalize: Whether to L2-normalize the output embeddings.
            show_progress: Whether to display a tqdm progress bar if available.
            progress_desc: Progress bar label.
            return_type: Whether to return a NumPy array or torch tensor.

        Returns:
            Embedding matrix aligned to the input text order.
        """
        if len(texts) == 0:
            empty = torch.empty((0, 0), dtype=torch.float32)
            return empty.numpy() if return_type == "numpy" else empty

        iterator = range(0, len(texts), batch_size)
        if show_progress:
            try:
                from tqdm.auto import tqdm

                iterator = tqdm(
                    iterator,
                    desc=progress_desc,
                    total=(len(texts) + batch_size - 1) // batch_size,
                )
            except ImportError:
                pass

        chunks = []
        for start in iterator:
            batch = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}

            output = self.model(**encoded)
            pooled = mean_pooling(output.last_hidden_state, encoded["attention_mask"])

            if normalize:
                pooled = F.normalize(pooled, p=2, dim=1)

            chunks.append(pooled.detach().cpu())

        embeddings = torch.cat(chunks, dim=0).to(torch.float32)
        if return_type == "torch":
            return embeddings
        return embeddings.numpy()
