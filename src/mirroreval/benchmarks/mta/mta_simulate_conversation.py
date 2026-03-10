"""
MTA Conversation Simulator
===========================
Simulates multi-turn conversations between a human and the model under test.

The MTA dataset contains an initial prompt and up to 3 follow-up messages.
This module feeds them to the model sequentially, building up a conversation
history so the model sees the full context at each turn — just like a real
chat interaction.

The output is a JSONL file where each line contains the original prompts
and all model responses for one conversation.
"""

import json

from mirroreval.config import settings
from mirroreval.hf_utilities import get_hf_pipeline, has_chat_template, get_hf_tokenizer


def simulate_conversation(dataset, output_file):
    """
    Run the model under test through every example in the dataset.

    Args:
        dataset: An iterable of dicts, each containing a "prompt" and optional
                 "followup_1", "followup_2", "followup_3" fields.
        output_file: Path to write the JSONL results to.

    Returns:
        List of result dicts (one per conversation).
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

    results = []

    for idx, example in enumerate(dataset):
        # --- Collect all user turns for this conversation ---
        # Each example has a "prompt" (required) and up to 3 follow-ups.
        # We gather them into a list to iterate through sequentially.
        user_turns = [example["prompt"]]
        for i in range(1, 4):
            key = f"followup_{i}"
            if key in example:
                user_turns.append(example[key])
            else:
                break

        responses = []

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
                    max_length=max_model_len,
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
                    conversation_text = tokenizer.decode(input_ids, skip_special_tokens=True)

                output = pipe(
                    conversation_text,
                    max_new_tokens=max_new_tokens,
                    max_length=max_model_len,
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
        results.append(
            {
                "example_id": idx,
                "prompt": example["prompt"],
                "followup_1": example.get("followup_1", None),
                "followup_2": example.get("followup_2", None),
                "followup_3": example.get("followup_3", None),
                "response_1": responses[0] if len(responses) > 0 else None,
                "response_2": responses[1] if len(responses) > 1 else None,
                "response_3": responses[2] if len(responses) > 2 else None,
                "response_4": responses[3] if len(responses) > 3 else None,
            }
        )

    # Write all results as JSONL (one JSON object per line).
    # Each line also includes the dataset class name for traceability.
    with open(output_file, "w") as f:
        for result in results:
            result["dataset"] = dataset.__class__.__name__
            f.write(json.dumps(result) + "\n")

    return results
