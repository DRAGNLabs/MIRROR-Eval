"""
Creativity Message Processing
=============================
Utilities for transforming multiturn conversation datasets into rows that can
be embedded by the creativity benchmark.

This module is the preprocessing counterpart to MIRROR-CAP's original
pipeline. It extracts numbered prompt/response turns, explodes conversations
into one row per turn, optionally splits each turn into sentences, and returns
datasets that preserve row and turn traceability for downstream metrics.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Optional, Tuple

from datasets import DatasetDict

from mirroreval.hf_utilities import load_hf_dataset


def extract_message_turns(
    example: Dict[str, Any],
    *,
    message_prefix: str,
    max_turns: int = 5,
    output_key: str = "message_turns",
    id_key: str = "message_turn_ids",
    keep_empty: bool = False,
) -> Dict[str, Any]:
    """
    Extract numbered message turns from a single raw dataset example.

    The benchmark expects columns like `P1`, `P2`, `R1`, `R2`, etc. This helper
    collects the selected prefix into an ordered list of turn texts plus the
    corresponding numeric turn ids.
    """
    turns: List[str] = []
    turn_ids: List[int] = []

    for i in range(1, max_turns + 1):
        value = example.get(f"{message_prefix}{i}")
        if value is None:
            continue

        text = str(value).strip()
        if not text and not keep_empty:
            continue

        turns.append(text)
        turn_ids.append(i)

    return {output_key: turns, id_key: turn_ids}


def explode_turns(
    batch: Dict[str, List[Any]],
    *,
    turns_key: str,
    turn_ids_key: str,
    text_out_key: str = "turn_text",
    turn_id_out_key: str = "turn_id",
    keep_row_id: bool = True,
    row_id_key: str = "row_id",
) -> Dict[str, List[Any]]:
    """
    Expand list-of-turn columns into one output row per turn.

    Intended for use with `datasets.Dataset.map(..., batched=True)`.
    """
    texts: List[str] = []
    turn_ids: List[int] = []
    row_ids: List[int] = []

    if keep_row_id and row_id_key not in batch:
        raise KeyError(
            f"keep_row_id=True but '{row_id_key}' not found in batch. "
            f"Add it first to the dataset."
        )

    for idx in range(len(batch[turns_key])):
        turns = batch[turns_key][idx]
        ids = batch[turn_ids_key][idx]
        row_id = batch[row_id_key][idx] if keep_row_id else None

        for text, turn_id in zip(turns, ids):
            texts.append(str(text))
            turn_ids.append(int(turn_id))
            if keep_row_id:
                row_ids.append(int(row_id))

    out: Dict[str, List[Any]] = {text_out_key: texts, turn_id_out_key: turn_ids}
    if keep_row_id:
        out[row_id_key] = row_ids
    return out


class SpacySentenceSplitter:
    """
    Sentence splitter with graceful fallback when spaCy is unavailable.

    If spaCy and a language pipeline are available, sentence segmentation uses
    the model's boundaries. Otherwise the splitter falls back to a simple regex
    strategy so the benchmark can still run in lighter environments.
    """

    def __init__(
        self,
        model_name: str = "en_core_web_sm",
        *,
        disable: Optional[List[str]] = None,
        max_length: int = 2_000_000,
    ) -> None:
        self.nlp = None
        try:
            import spacy

            disable = disable or [
                "ner",
                "tagger",
                "lemmatizer",
                "attribute_ruler",
            ]
            try:
                self.nlp = spacy.load(model_name, disable=disable)
            except OSError:
                self.nlp = spacy.blank("en")
                if "sentencizer" not in self.nlp.pipe_names:
                    self.nlp.add_pipe("sentencizer")
            self.nlp.max_length = max_length
        except ImportError:
            self.nlp = None

    def split(
        self,
        text: str,
        *,
        min_chars: int = 3,
        normalize_whitespace: bool = True,
        drop_code_fences: bool = False,
    ) -> List[str]:
        """
        Split a single turn into filtered sentence strings.

        Args:
            text: Turn text to segment.
            min_chars: Minimum length for retained sentences after stripping.
            normalize_whitespace: Whether to collapse repeated whitespace.
            drop_code_fences: Whether to remove fenced code blocks before
                splitting.

        Returns:
            A list of cleaned sentence strings.
        """
        if not text:
            return []

        value = text.strip()

        if drop_code_fences:
            value = re.sub(r"```.*?```", " ", value, flags=re.DOTALL)

        if normalize_whitespace:
            value = re.sub(r"\s+", " ", value).strip()

        if not value:
            return []

        if self.nlp is None:
            sentences = re.split(r"(?<=[.!?])\s+", value)
        else:
            doc = self.nlp(value)
            sentences = [sentence.text.strip() for sentence in doc.sents]

        return [sentence for sentence in sentences if len(sentence) >= min_chars]


def explode_sentences(
    batch: Dict[str, List[Any]],
    *,
    sentences_key: str = "sentences",
    sentence_out_key: str = "sentence_text",
    keep_keys: Optional[List[str]] = None,
    drop_empty: bool = True,
    min_chars: int = 1,
) -> Dict[str, List[Any]]:
    """
    Expand sentence-list columns into one output row per sentence.

    This is the final structural step for `mode="sentence"`, producing rows
    that can be embedded independently while still retaining the source row and
    turn identifiers.
    """
    keep_keys = keep_keys or []

    out: Dict[str, List[Any]] = {sentence_out_key: []}
    for key in keep_keys:
        out[key] = []

    for idx in range(len(batch[sentences_key])):
        sentences = batch[sentences_key][idx] or []
        if not sentences:
            continue
        if isinstance(sentences, str):
            sentences = [sentences]

        for sentence in sentences:
            cleaned = str(sentence).strip()
            if drop_empty and not cleaned:
                continue
            if len(cleaned) < min_chars:
                continue

            out[sentence_out_key].append(cleaned)
            for key in keep_keys:
                out[key].append(batch[key][idx])

    return out


def add_turn_columns(
    dataset: DatasetDict,
    *,
    message_prefix: str,
    output_key: str,
    id_key: str,
    max_turns: int = 5,
) -> DatasetDict:
    """Add extracted turn-text and turn-id list columns to each example."""
    return dataset.map(
        extract_message_turns,
        fn_kwargs={
            "message_prefix": message_prefix,
            "output_key": output_key,
            "id_key": id_key,
            "max_turns": max_turns,
        },
    )


def load_multi_turn_dataset(
    dataset_name: str = "jackwarner/multi-turn-conversations",
) -> DatasetDict:
    """
    Load and normalize the base multiturn conversation dataset.

    The benchmark expects the schema introduced in MIRROR-CAP:
      - an `ID` column in the raw dataset
      - numbered turn columns like `P1`/`R1`
      - a stable `row_id` column added if one is not already present

    Some metadata columns are removed because they are not needed for the
    benchmark pipeline and only increase the cost of later dataset maps.
    """
    dataset = load_hf_dataset(dataset_name)

    train_cols = set(dataset["train"].column_names)
    if "ID" not in train_cols:
        raise ValueError(
            "Dataset must include an 'ID' column. "
            "Expected schema from 'jackwarner/multi-turn-conversations'."
        )

    if "row_id" not in train_cols:
        dataset = dataset.map(lambda _, idx: {"row_id": idx}, with_indices=True)

    removable = [
        column
        for column in ("Use case", "Type", "Category")
        if column in set(dataset["train"].column_names)
    ]
    if removable:
        dataset = dataset.remove_columns(removable)
    return dataset


def explode_turn_dataset(
    dataset: DatasetDict,
    *,
    turns_key: str,
    turn_ids_key: str,
    text_out_key: str,
    turn_id_out_key: str,
    row_id_key: str = "row_id",
) -> DatasetDict:
    """
    Convert extracted turn lists into a dataset with one row per turn.

    The returned dataset retains `row_id` so later metric computation can
    regroup exploded rows back into conversations.
    """
    return dataset.map(
        explode_turns,
        batched=True,
        fn_kwargs={
            "turns_key": turns_key,
            "turn_ids_key": turn_ids_key,
            "text_out_key": text_out_key,
            "turn_id_out_key": turn_id_out_key,
            "keep_row_id": True,
            "row_id_key": row_id_key,
        },
        remove_columns=dataset["train"].column_names,
    )


def split_turns_to_sentences(
    turn_dataset: DatasetDict,
    *,
    turn_text_key: str,
    splitter: Optional[SpacySentenceSplitter] = None,
) -> DatasetDict:
    """Add a `sentences` list column to each turn row."""
    sentence_splitter = splitter or SpacySentenceSplitter()
    return turn_dataset.map(
        lambda example: {"sentences": sentence_splitter.split(example[turn_text_key])},
    )


def explode_sentence_dataset(
    sentence_dataset: DatasetDict,
    *,
    sentence_out_key: str,
    keep_keys: List[str],
) -> DatasetDict:
    """Convert sentence-list rows into a dataset with one row per sentence."""
    return sentence_dataset.map(
        explode_sentences,
        batched=True,
        fn_kwargs={
            "sentences_key": "sentences",
            "sentence_out_key": sentence_out_key,
            "keep_keys": keep_keys,
        },
        remove_columns=sentence_dataset["train"].column_names,
    )


def build_sentence_rows_for_role(
    dataset: DatasetDict,
    *,
    message_prefix: str,
    turns_key: str,
    turn_ids_key: str,
    turn_text_out_key: str,
    turn_id_out_key: str,
    sentence_out_key: str,
    row_id_key: str = "row_id",
    splitter: Optional[SpacySentenceSplitter] = None,
) -> DatasetDict:
    """
    Run the full sentence-mode preprocessing pipeline for one conversation role.

    Pipeline:
      1. Extract role-specific numbered turns
      2. Explode to one row per turn
      3. Split each turn into sentences
      4. Explode to one row per sentence
    """
    with_turns = add_turn_columns(
        dataset,
        message_prefix=message_prefix,
        output_key=turns_key,
        id_key=turn_ids_key,
    )
    turn_rows = explode_turn_dataset(
        with_turns,
        turns_key=turns_key,
        turn_ids_key=turn_ids_key,
        text_out_key=turn_text_out_key,
        turn_id_out_key=turn_id_out_key,
        row_id_key=row_id_key,
    )
    with_sentences = split_turns_to_sentences(
        turn_rows,
        turn_text_key=turn_text_out_key,
        splitter=splitter,
    )
    return explode_sentence_dataset(
        with_sentences,
        sentence_out_key=sentence_out_key,
        keep_keys=[row_id_key, turn_id_out_key],
    )


def build_turn_rows_for_role(
    dataset: DatasetDict,
    *,
    message_prefix: str,
    turns_key: str,
    turn_ids_key: str,
    turn_text_out_key: str,
    turn_id_out_key: str,
    row_id_key: str = "row_id",
) -> DatasetDict:
    """
    Build one-row-per-turn data for the selected role.

    This is the preprocessing path for `mode="message"`, where each turn is
    embedded as a single vector instead of being split into sentences first.
    """
    with_turns = add_turn_columns(
        dataset,
        message_prefix=message_prefix,
        output_key=turns_key,
        id_key=turn_ids_key,
    )
    return explode_turn_dataset(
        with_turns,
        turns_key=turns_key,
        turn_ids_key=turn_ids_key,
        text_out_key=turn_text_out_key,
        turn_id_out_key=turn_id_out_key,
        row_id_key=row_id_key,
    )
