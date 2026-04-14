"""
MQM Datasets
============
Defines dataset classes for the MT Metric Reliability (MQM) benchmark.

Two datasets are registered:
  - "flores-plus": FLORES+ multilingual sentences for translating with the
    model under test. Yields (source, reference, lang) triples.
  - "wmt-da-calibration": WMT Direct Assessment data used only to compute
    metric reliability calibration numbers. Not used at inference time.

The FLORES+ dataset is the primary input: the model under test translates
English source sentences into each configured target language, and automated
MT metrics are run against the FLORES+ reference translations.
"""

from typing import Any, Iterator

from mirroreval.benchmarks.interfaces import DatasetInterface, register_dataset
from mirroreval.hf_utilities import load_hf_dataset
from mirroreval.logger import logger

# Maps 2-letter ISO codes (used throughout this benchmark) to FLORES+ language
# codes (used as column names in the openlanguagedata/flores_plus dataset).
FLORES_LANG_CODES = {
    "de": "deu_Latn",
    "zh": "zho_Hans",
    "ru": "rus_Cyrl",
    "he": "heb_Hebr",
    "es": "spa_Latn",
    "cs": "ces_Latn",
    "tr": "tur_Latn",
    "uk": "ukr_Cyrl",
    "ha": "hau_Latn",
    "km":  "khm_Khmr",
    "ps": "pbt_Arab",
    "sw": "swh_Latn",
    "ht": "hat_Latn",
    "lo": "lao_Laoo",
}

# NLLB-200 language tags for seq2seq translation pipelines.
# The model under test is expected to be an NLLB-200-style model that accepts
# these src_lang / tgt_lang tokens.
NLLB_LANG_CODES = {
    "de": "deu_Latn",
    "zh": "zho_Hans",
    "ru": "rus_Cyrl",
    "he": "heb_Hebr",
    "es": "spa_Latn",
    "cs": "ces_Latn",
    "tr": "tur_Latn",
    "uk": "ukr_Cyrl",
    "ha": "hau_Latn",
    "km": "khm_Khmr",
    "ps": "pbt_Arab",
    "sw": "swh_Latn",
    "ht": "hat_Latn",
    "lo": "lao_Laoo",
}


@register_dataset("openlanguagedata/flores_plus")
class FLORESDataset(DatasetInterface):
    """
    FLORES+ multilingual evaluation set.

    Yields dicts with:
      - "source": English source sentence
      - "reference": reference translation in the target language
      - "lang": 2-letter ISO code for the target language
      - "flores_lang": FLORES+ language code (e.g., "deu_Latn")
      - "nllb_lang": NLLB-200 language tag for the target language
      - "segment_id": integer position within the dataset

    The FLORESDataset is constructed with a list of target language codes
    (2-letter ISO) to determine which languages to yield.
    """

    def __init__(self, target_langs: list[str], split: str = "devtest"):
        self.target_langs = target_langs
        self.split = split
        self.dataset = None
        self.load_data()

    def load_data(self) -> None:
        self.dataset = load_hf_dataset("openlanguagedata/flores_plus")
        logger.info(
            f"Loaded FLORES+ ({self.split}), "
            f"targeting languages: {self.target_langs}"
        )

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if self.dataset is None:
            raise ValueError("Dataset not loaded.")
        split_data = self.dataset[self.split]
        eng_col = "eng_Latn"
        for lang in self.target_langs:
            flores_lang = FLORES_LANG_CODES.get(lang)
            if flores_lang is None:
                logger.warning(f"No FLORES+ code for lang '{lang}', skipping.")
                continue
            if flores_lang not in split_data.column_names:
                logger.warning(
                    f"FLORES+ split '{self.split}' has no column '{flores_lang}', skipping."
                )
                continue
            for idx, row in enumerate(split_data):
                yield {
                    "segment_id": idx,
                    "lang": lang,
                    "flores_lang": flores_lang,
                    "nllb_lang": NLLB_LANG_CODES.get(lang, flores_lang),
                    "source": row[eng_col],
                    "reference": row[flores_lang],
                }

    def __len__(self) -> int:
        if self.dataset is None:
            return 0
        return len(self.dataset[self.split]) * len(self.target_langs)
