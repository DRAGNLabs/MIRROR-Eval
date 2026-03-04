import json

from mirroreval.config import settings
from mirroreval.hf_utilities import get_hf_pipeline


def simulate_conversation(dataset, output_file):
    """
    Simulate multi-turn conversations between a human and a model.
    Each turn feeds the full conversation history to the model sequentially.
    """

    model_name = settings.model.model_checkpoint_path

    # TODO: we assume this is a HF model for now, but what if we wanted to load locally. Need to add the option
    pipe = get_hf_pipeline(model_name)

    results = []

    for idx, example in enumerate(dataset):
        # Collect user turns: initial prompt + any followups
        user_turns = [example["prompt"]]
        for i in range(1, 4):
            key = f"followup_{i}"
            if key in example:
                user_turns.append(example[key])
            else:
                break

        conversation = []
        responses = []

        for user_msg in user_turns:
            conversation.append({"role": "user", "content": user_msg})

            output = pipe(conversation, max_new_tokens=256)
            assistant_msg = output[0]["generated_text"][-1]["content"]

            conversation.append({"role": "assistant", "content": assistant_msg})

            responses.append(assistant_msg)

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

    # Write results as JSONL
    with open(output_file, "w") as f:
        for result in results:
            result["dataset"] = dataset.__class__.__name__
            f.write(json.dumps(result) + "\n")

    return results
