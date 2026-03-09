import numpy as np
import json
import os
import tempfile
import torch
from sklearn.metrics.pairwise import cosine_distances
from typing import List, Any, Iterator

from mirroreval.benchmarks.interfaces import MetricInterface, register_metric
from mirroreval.config import settings
from mirroreval.benchmarks.mta.prompts import (
    get_prompt_names,
    get_formatted_prompt,
)
from mirroreval.hf_utilities import get_hf_pipeline, get_hf_model, get_hf_tokenizer
from mirroreval.logger import logger

BATCH_SIZE = 128  # Adjust based on your model/GPU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@register_metric("llm-as-a-judge")
class LLMAsAJudge(MetricInterface):

    def chunked(self, iterable, n):
        """Yield successive n-sized chunks from iterable."""
        for i in range(0, len(iterable), n):
            yield iterable[i : i + n]

    def __call__(self, dataset):
        """
        Takes a dataset generator and yields a dictionary with the following keys:
        - input_id: The index of the input line in the dataset
        - split: The dataset split (e.g., train, validation, test)
        - metric: The name of the metric ("llm-as-a-judge")
        - model_name: The name of the model used
        - prompt: The prompt name used
        - output: The raw output from the model
        - score: The score assigned by the model (0 for set1 better, 1
                    for set2 better, -1 for tie)
        - accuracy: 1 if the model's score matches the ground truth, 0 otherwise
        """
        judge_model = settings.mta.llm_judge_model
        judge_prompt_name = settings.mta.judge_prompt_name

        pipeline = get_hf_pipeline(judge_model)

        sample_chunk = []

        def flush_chunk(chunk, out_f):
            prompt_chunk = [sample[2] for sample in chunk]
            outputs = pipeline(prompt_chunk, max_new_tokens=256, max_length=None, num_return_sequences=1)
            outputs = [output[0]["generated_text"][-1]["content"] for output in outputs]
            scores = []
            # TODO: what if the output is malformed?
            for output in outputs:
                # Parse output to JSON to get score
                try:
                    output_json = json.loads(output)
                    score = output_json.get("score", None)
                    if score is not None:
                        scores.append(score)
                    else:
                        logger.warning(
                            f"Output JSON does not contain 'score' field: {output}"
                        )
                        scores.append(None)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse output as JSON: {output}")
                    scores.append(None)

            for sample, output, score in zip(chunk, outputs, scores):
                input_line = sample[1]
                input_line.update(
                    {
                        "prompt": sample[2],
                        "output": output,
                        "llm_as_a_judge_score": score,
                    }
                )
                out_f.write(json.dumps(input_line) + "\n")

        tmp_path = dataset.with_suffix(dataset.suffix + ".tmp")
        with open(dataset, "r") as in_f, open(tmp_path, "w") as out_f:
            for index, raw_line in enumerate(in_f):
                input_line = json.loads(raw_line)
                logger.info(f"Processing input line {index}")

                formatted_prompt = get_formatted_prompt(
                    model_name=judge_model,
                    prompt_name=judge_prompt_name,
                    prompt=input_line["prompt"],
                    response=input_line["response_4"],
                )

                sample_chunk.append((index, input_line, formatted_prompt))

                if len(sample_chunk) >= BATCH_SIZE:
                    flush_chunk(sample_chunk, out_f)
                    sample_chunk = []

            # Flush remaining samples that didn't fill a full batch
            if sample_chunk:
                flush_chunk(sample_chunk, out_f)

        os.replace(tmp_path, dataset)
