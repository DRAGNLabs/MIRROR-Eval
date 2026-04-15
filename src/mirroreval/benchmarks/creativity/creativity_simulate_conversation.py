"""
Creativity Conversation Simulator
=================================
Simulates multi-turn conversations for the creativity benchmark.

This module mirrors MTA's conversation-generation behavior: it consumes an
iterable dataset, normalizes each example into a `prompt` plus ordered
`followup_n` user turns, runs the model under test sequentially across those
turns, and returns a generated multiturn dataset with numbered `P*` and `R*`
columns for the downstream embedding pipeline.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict

from mirroreval.config import settings
from mirroreval.hf_utilities import (
    get_hf_pipeline,
    get_hf_tokenizer,
    has_chat_template,
)


def _normalize_example_to_user_turns(
    example: dict[str, Any],
    *,
    max_turns: int = 5,
) -> list[str]:
    """
    Convert a raw creativity example into ordered user turns.

    Supported input schemas:
      - MTA-style fields: `prompt`, `followup_1`, `followup_2`, ...
      - Creativity-style numbered prompt fields: `P1`, `P2`, ...
      - Archetype lists under `archetypes`
      - Single values under `archetype`
    """
    # Here's where the logic in lines 59-65 of the mta counterpart is implemented.
    # Creativity can normalize multiple schemas, and use the hardcoded dataset I made, or generate one.
    if "prompt" in example:
        turns = [str(example["prompt"]).strip()]
        for turn_id in range(1, max_turns):
            key = f"followup_{turn_id}"
            value = example.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                turns.append(text)
        return [turn for turn in turns if turn]

    numbered_prompt_keys = []
    for key in example:
        match = re.fullmatch(r"P(\d+)", key)
        if match:
            numbered_prompt_keys.append((int(match.group(1)), key))

    if numbered_prompt_keys:
        turns = []
        for _, key in sorted(numbered_prompt_keys):
            text = str(example[key]).strip()
            if text:
                turns.append(text)
        return turns[:max_turns]

    if "archetypes" in example:
        archetypes = example["archetypes"]
        if isinstance(archetypes, str):
            turns = [archetypes.strip()]
        else:
            turns = [str(value).strip() for value in archetypes if str(value).strip()]
        return turns[:max_turns]

    if "archetype" in example:
        text = str(example["archetype"]).strip()
        return [text] if text else []

    raise ValueError(
        "Creativity examples must provide one of: "
        "`prompt`/`followup_n`, numbered `Pn` fields, `archetypes`, or `archetype`."
    )


def simulate_conversation(
    dataset,
    output_file: Path | None = None,
    *,
    max_turns: int = 5,
) -> DatasetDict:
    """
    Run the model under test through every example in the dataset.

    Args:
        dataset: Iterable of examples describing user-turn archetypes.
        output_file: Optional JSONL path for writing MTA-style generated
            conversations (`prompt`, `followup_n`, `response_n`).
        max_turns: Maximum number of user turns to simulate per example.

    Returns:
        DatasetDict with a `"train"` split containing `ID`, `row_id`, and
        generated numbered `P*` and `R*` columns.
    """
    model_name = settings.model.model_checkpoint_path

    # Create a HuggingFace text-generation pipeline for the model under test.
    # TODO: we assume this is a HF model for now, but what if we wanted to load locally. Need to add the option
    pipe = get_hf_pipeline(model_name)

    # Determine whether the model supports chat templates (role/content dicts).
    # Models like Llama-Instruct or Qwen-Chat have chat templates; base models
    # like distilgpt2 do not. This flag controls how we format inputs and
    # parse outputs for the entire function.
    use_chat = has_chat_template(model_name)

    # Load the tokenizer to measure token counts for context window management.
    # model_max_length is the model's maximum context window (e.g., 1024 for
    # distilgpt2, 8192 for Llama). We reserve max_new_tokens for generation.
    tokenizer = get_hf_tokenizer(model_name)
    max_model_len = getattr(tokenizer, "model_max_length", 1024)
    max_new_tokens = 256

    generated_records: list[dict[str, Any]] = []
    jsonl_records: list[dict[str, Any]] = []

    for idx, example in enumerate(dataset):
        # --- Collect all user turns for this conversation ---
        # Each example has a "prompt" (required) and up to 3 follow-ups.
        # We gather them into a list to iterate through sequentially.
        user_turns = _normalize_example_to_user_turns(example, max_turns=max_turns)
        
        responses: list[str] = []

        if use_chat:
            # --- Chat-template path ---
            # For models with chat templates, we pass a list of role/content
            # dicts. The HF pipeline applies the model's chat template
            # internally (via tokenizer.apply_chat_template) to format the
            # conversation correctly for that specific model.
            #
            # The conversation grows with each turn: we append the user
            # message, generate a response, then append the assistant message
            # before the next turn. This gives the model full history.
            #
            # Output format: output[0]["generated_text"] is a list of message
            # dicts. The last element is the assistant's new response.
            conversation = []
            for user_msg in user_turns:
                conversation.append({"role": "user", "content": user_msg})
                output = pipe(
                    conversation,
                    max_new_tokens=max_new_tokens,
                )
                assistant_msg = output[0]["generated_text"][-1]["content"]
                conversation.append({"role": "assistant", "content": assistant_msg})
                responses.append(assistant_msg)
        else:
            # --- Plain text path ---
            # For models without chat templates (e.g., distilgpt2), we format
            # the conversation as a plain string with "User:" and "Assistant:"
            # prefixes. The model sees this as a text completion task.
            #
            # Output format: output[0]["generated_text"] is a single string
            # containing the full input + generated text. We strip the input
            # prefix to isolate just the model's response.
            conversation_text = ""
            for user_msg in user_turns:
                conversation_text += f"User: {user_msg}\nAssistant:"

                # Context window management: if the accumulated conversation
                # exceeds the model's context window (minus room for generation),
                # truncate from the LEFT. This preserves the most recent context
                # at the cost of losing early conversation history.
                input_ids = tokenizer.encode(conversation_text)
                budget = max_model_len - max_new_tokens
                if len(input_ids) > budget:
                    input_ids = input_ids[-budget:]
                    conversation_text = tokenizer.decode(
                        input_ids,
                        skip_special_tokens=True,
                    )

                output = pipe(
                    conversation_text,
                    max_new_tokens=max_new_tokens,
                )
                full_text = output[0]["generated_text"]

                # Extract only the newly generated text by stripping the input.
                assistant_msg = full_text[len(conversation_text) :].strip()


                # Base models don't know when to stop — they may generate the
                # next "User:" turn themselves. Truncate at the first occurrence
                # to keep only the assistant's actual response.
                if "User:" in assistant_msg:
                    assistant_msg = assistant_msg[: assistant_msg.index("User:")].strip()

                conversation_text += f" {assistant_msg}\n"
                responses.append(assistant_msg)

        # Store all prompts and responses for this conversation.
        # response_1 corresponds to the initial prompt, response_2 to followup_1, etc.
        generated_record: dict[str, Any] = {
            "ID": example.get("ID", idx),
            "row_id": example.get("row_id", idx),
        }
        jsonl_record: dict[str, Any] = {
            "example_id": idx,
            "prompt": user_turns[0] if user_turns else None,
            "dataset": dataset.__class__.__name__,
        }

        for turn_idx, user_msg in enumerate(user_turns, start=1):
            generated_record[f"P{turn_idx}"] = user_msg
            if turn_idx == 1:
                jsonl_record["prompt"] = user_msg
            else:
                jsonl_record[f"followup_{turn_idx - 1}"] = user_msg

        for turn_idx, assistant_msg in enumerate(responses, start=1):
            generated_record[f"R{turn_idx}"] = assistant_msg
            jsonl_record[f"response_{turn_idx}"] = assistant_msg

        generated_records.append(generated_record)
        jsonl_records.append(jsonl_record)

    if output_file is not None:
        with open(output_file, "w", encoding="utf-8") as handle:
            for record in jsonl_records:
                handle.write(json.dumps(record) + "\n")

    return DatasetDict({"train": Dataset.from_list(generated_records)})
