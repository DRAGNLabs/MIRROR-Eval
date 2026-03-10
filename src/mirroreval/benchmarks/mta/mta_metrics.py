"""
MTA Metrics — LLM-as-a-Judge
==============================
Implements the LLM-as-a-judge scoring metric for MTA.

This metric uses a separate "judge" LLM to evaluate whether the model under
test retained information across a multi-turn conversation. The judge reads
the original prompt and the model's final response, then assigns a score.

The metric operates on the intermediate JSONL file produced by
simulate_conversation. It reads each line, sends the prompt+response to the
judge LLM, parses the score from the judge's JSON output, and writes the
enriched records back to the same file (atomically via temp file + replace).

Registration:
  The @register_metric("llm-as-a-judge") decorator adds this class to the
  global METRICS dict. It is discovered at import time via the side-effect
  import in mta_benchmark.py.
"""

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
from mirroreval.hf_utilities import get_hf_pipeline, get_hf_model, get_hf_tokenizer, has_chat_template
from mirroreval.logger import logger

BATCH_SIZE = 128  # Adjust based on your model/GPU memory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@register_metric("llm-as-a-judge")
class LLMAsAJudge(MetricInterface):

    def chunked(self, iterable, n):
        """Yield successive n-sized chunks from iterable."""
        for i in range(0, len(iterable), n):
            yield iterable[i : i + n]

    def __call__(self, dataset):
        """
        Score every conversation in the JSONL file using an LLM judge.

        Args:
            dataset: A pathlib.Path to the intermediate JSONL file containing
                     model responses from simulate_conversation.

        The method reads the file, sends each prompt+response pair to the judge
        LLM in batches, parses scores from the judge's output, and writes the
        updated records (now including "llm_as_a_judge_score") back to the file.
        """
        # --- Configuration ---
        # The judge model and prompt name are specified in settings.toml under [mta].
        # judge_model: HuggingFace model ID for the judge (e.g., "distilgpt2")
        # judge_prompt_name: which prompt template to use (e.g., "scale" for 1-7 scoring)
        judge_model = settings.mta.llm_judge_model
        judge_prompt_name = settings.mta.judge_prompt_name

        # Create a text-generation pipeline for the judge model.
        pipe = get_hf_pipeline(judge_model)

        # Check if the judge model supports chat templates. This determines how
        # we format prompts (chat dicts vs plain text) and parse outputs.
        use_chat = has_chat_template(judge_model)

        # Load tokenizer for context window management (same pattern as
        # simulate_conversation — see that module for detailed explanation).
        tokenizer = get_hf_tokenizer(judge_model)
        max_model_len = getattr(tokenizer, "model_max_length", 1024)
        max_new_tokens = 256

        # Accumulates samples until we have a full batch to send to the model.
        sample_chunk = []

        def extract_outputs(raw_outputs, prompt_chunk):
            """
            Extract the generated text from pipeline output.

            Chat models return a list of message dicts — we grab the last
            message's content. Plain text models return a string that includes
            the input — we strip the input prefix to get only the generated part.
            """
            if use_chat:
                return [out[0]["generated_text"][-1]["content"] for out in raw_outputs]
            else:
                extracted = []
                for out, prompt in zip(raw_outputs, prompt_chunk):
                    text = out[0]["generated_text"]
                    # Strip the input prompt to get only the generated portion
                    generated = text[len(prompt):].strip() if isinstance(prompt, str) else text
                    extracted.append(generated)
                return extracted

        def flush_chunk(chunk, out_f):
            """
            Process a batch of samples through the judge model and write results.

            Each sample in the chunk is a tuple: (index, original_line_dict, formatted_prompt).
            We extract the prompts, run them through the pipeline in one batch,
            parse the JSON scores from each output, and write enriched records.
            """
            prompt_chunk = [sample[2] for sample in chunk]

            # For plain text models, truncate prompts that exceed the context
            # window. We keep the rightmost tokens (most relevant context).
            if not use_chat:
                budget = max_model_len - max_new_tokens
                truncated = []
                for p in prompt_chunk:
                    ids = tokenizer.encode(p)
                    if len(ids) > budget:
                        ids = ids[-budget:]
                        p = tokenizer.decode(ids, skip_special_tokens=True)
                    truncated.append(p)
                prompt_chunk = truncated

            # Run the entire batch through the judge model at once.
            # Batching is significantly faster than individual calls.
            raw_outputs = pipe(prompt_chunk, max_new_tokens=max_new_tokens, max_length=max_model_len, num_return_sequences=1)
            outputs = extract_outputs(raw_outputs, prompt_chunk)

            # --- Parse scores from judge output ---
            # The judge is instructed to return JSON like {"score": 5}.
            # In practice (especially with small/base models), the output may
            # not be valid JSON — we handle that gracefully with None scores.
            scores = []
            for output in outputs:
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

            # Write each enriched record to the output file.
            # We add the formatted prompt, raw judge output, and parsed score
            # to the original record from the JSONL file.
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

        # --- Process the JSONL file ---
        # We read from the original file and write to a temp file. Once all
        # records are processed, we atomically replace the original with
        # os.replace(). This avoids corrupting the file if the process crashes
        # mid-write.
        tmp_path = dataset.with_suffix(dataset.suffix + ".tmp")
        with open(dataset, "r") as in_f, open(tmp_path, "w") as out_f:
            for index, raw_line in enumerate(in_f):
                input_line = json.loads(raw_line)
                logger.info(f"Processing input line {index}")

                # Build the judge prompt from the template. This combines the
                # prompt template (e.g., "scale") with the actual prompt and
                # the model's final response (response_4 = last turn).
                formatted_prompt = get_formatted_prompt(
                    model_name=judge_model,
                    prompt_name=judge_prompt_name,
                    use_chat=use_chat,
                    prompt=input_line["prompt"],
                    response=input_line["response_4"],
                )

                # Accumulate samples into a batch. Each sample is a tuple of
                # (line_index, original_record, formatted_prompt).
                sample_chunk.append((index, input_line, formatted_prompt))

                # When we have a full batch, flush it through the model.
                if len(sample_chunk) >= BATCH_SIZE:
                    flush_chunk(sample_chunk, out_f)
                    sample_chunk = []

            # Don't forget the last partial batch.
            if sample_chunk:
                flush_chunk(sample_chunk, out_f)

        # Atomically replace the original file with the enriched version.
        os.replace(tmp_path, dataset)
