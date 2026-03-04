import copy

PROMPT_NAMES = ["scale", "category"]

PROMPTS = {
    # Detailed source: [Evaluating the Evaluation of Diversity in Commonsense Generation](https://aclanthology.org/2025.acl-long.1181/) (Zhang et al., ACL 2025)
    # Modified to ellict more correct responses. Adds more details on formatting.
    "scale": """
    Rate, on a scale of 1 to 10, whether the information presented in the prompt is retained in the final response.

    Prompt: {prompt}
    Response: {response}

    Please return in the JSON format: {{"score": <score>}} where <score> is an integer from 1 to 10, with 1 indicating that the information is not retained at all and 10 indicating that the information is fully retained.
    """,
    # Detailed with 5-shot examples
    "category": """
    Is the information presented in the prompt retained in the final response? Answer "Yes" if the information is retained, and "No" if it is not.

    Prompt: {prompt}
    Response: {response}

    Please return in the JSON format: {{"score": <score>}} where <score> is either "Yes" or "No".
    """,
}


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


def get_formatted_prompt(model_name, prompt_name, **kwargs):
    # Deep copy to avoid mutating the original
    prompt_template = copy.deepcopy(SYSTEM_PROMPTS[model_name])
    actual_prompt = PROMPTS[prompt_name].format(**kwargs)
    for message in prompt_template:
        if message["role"] == "user":
            message["content"] = message["content"].format(prompt=actual_prompt)
    return prompt_template
