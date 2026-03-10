"""
MTA Prompts — Prompt Templates and Formatting
================================================
Centralizes all prompt templates used by the MTA benchmark.

Two types of prompts live here:
  1. PROMPTS: The actual evaluation instructions sent to the judge LLM.
     These contain {prompt} and {response} placeholders that get filled
     with the original user prompt and the model's response.
  2. SYSTEM_PROMPTS: Model-specific conversation formats for chat models.
     These define the system message and message structure expected by
     each supported model.

The get_formatted_prompt() function combines these, returning either
chat-formatted dicts (for models with chat templates) or a plain text
string (for base models like distilgpt2).

To add a new prompt variant:
  1. Add an entry to PROMPTS with a descriptive key
  2. Add the key to PROMPT_NAMES
  3. Use it by setting judge_prompt_name in settings.toml

To add support for a new chat model:
  1. Add an entry to SYSTEM_PROMPTS keyed by the model's HuggingFace ID
"""

import copy

# Available prompt template names. These are the valid values for
# settings.mta.judge_prompt_name in settings.toml.
PROMPT_NAMES = ["scale", "category"]

# Judge prompt templates. Each template instructs the judge LLM on how to
# evaluate the model's response. The {prompt} and {response} placeholders
# are filled at runtime by get_formatted_prompt().
#
# Double braces {{ }} are Python string escapes — they produce literal braces
# in the output (e.g., {{"score": <score>}} becomes {"score": <score>}).
PROMPTS = {
    # "scale": Judge assigns a score from 1 to 7.
    # Modified to elicit more correct responses with stricter formatting.
    "scale": """
    Rate, on a scale of 1 to 7, whether the information presented in the prompt is retained in the final response.

    Prompt: {prompt}
    Response: {response}

    Please return in the JSON format: {{"score": <score>}} where <score> is an integer from 1 to 7, with 1 indicating that the information is not retained at all and 7 indicating that the information is fully retained. DON'T return anything other than the JSON object with the score. DO NOT include any explanations or additional text in your response.
    """,
    # "category": Judge assigns a binary Yes/No.
    "category": """
    Is the information presented in the prompt retained in the final response? Answer "Yes" if the information is retained, and "No" if it is not.

    Prompt: {prompt}
    Response: {response}

    Please return in the JSON format: {{"score": <score>}} where <score> is either "Yes" or "No".
    """,
}


# Model-specific conversation formats for chat models.
# Each entry is a list of message dicts that the HF pipeline will format
# using the model's chat template. The {prompt} placeholder in the user
# message gets replaced with the actual formatted evaluation prompt.
#
# If a model is not listed here and has no chat template, the system
# falls back to plain text mode automatically.
SYSTEM_PROMPTS = {
    "meta-llama/Llama-3.3-70B-Instruct": [
        {
            "role": "system",
            "content": "You are a chat bot that answers directions",
        },
        {"role": "user", "content": "{prompt}"},
    ],
    "meta-llama/Meta-Llama-3.1-8B-Instruct": [
        {
            "role": "system",
            "content": "You are a chat bot that answers directions",
        },
        {"role": "user", "content": "{prompt}"},
    ],
    "Qwen/Qwen3-30B-A3B-Instruct-2507": [
        {
            "role": "system",
            "content": "You are a chat bot that answers directions",
        },
        {"role": "user", "content": "{prompt}"},
    ],
    "Qwen/Qwen1.5-0.5B-Chat": [
        {
            "role": "system",
            "content": "You are a chat bot that answers directions",
        },
        {"role": "user", "content": "{prompt}"},
    ],
    "Qwen/Qwen3-0.6B": [
        {
            "role": "system",
            "content": "You are a chat bot that answers directions",
        },
        {"role": "user", "content": "{prompt}"},
    ],
}


def get_prompt_names():
    return PROMPT_NAMES


def get_formatted_prompt(model_name, prompt_name, use_chat=True, **kwargs):
    """
    Build a formatted prompt for the judge LLM.

    Args:
        model_name: HuggingFace model ID (used to look up SYSTEM_PROMPTS).
        prompt_name: Key into PROMPTS (e.g., "scale" or "category").
        use_chat: If True and the model has an entry in SYSTEM_PROMPTS,
                  return chat-formatted dicts. Otherwise return plain text.
        **kwargs: Values to fill into the prompt template (prompt, response).

    Returns:
        Either a list of message dicts (chat mode) or a plain string (text mode).
    """
    # Format the evaluation prompt with the actual prompt and response values.
    actual_prompt = PROMPTS[prompt_name].format(**kwargs)

    if use_chat and model_name in SYSTEM_PROMPTS:
        # Deep copy prevents mutating the shared SYSTEM_PROMPTS template.
        # Without this, the first call would permanently replace {prompt}
        # and subsequent calls would fail or use stale data.
        prompt_template = copy.deepcopy(SYSTEM_PROMPTS[model_name])
        for message in prompt_template:
            if message["role"] == "user":
                message["content"] = message["content"].format(prompt=actual_prompt)
        return prompt_template

    # Plain text fallback — just return the evaluation prompt as a string.
    # The pipeline will treat this as a text completion task.
    return actual_prompt
