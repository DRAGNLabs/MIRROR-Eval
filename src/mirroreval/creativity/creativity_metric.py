import json
import sys

from mirroreval.config import init_settings, settings
from mirroreval.creativity.prompts import get_prompt_names, get_formatted_prompt
from mirroreval.hf_utilities import get_hf_pipeline, load_hf_dataset


def run_metric():
    dataset = load_hf_dataset("royal42/gcr-diversity")

    models = settings.creativity.models

    prompt_keys = get_prompt_names()

    for model_name in models:
        pipeline = get_hf_pipeline(model_name)
        for split_name, split_dataset in dataset.items():
            print(f"--- Split: {split_name} ---")
            print(f"Number of examples: {len(split_dataset)}")
            # Iterate through dataset
            for input_line in split_dataset:
                for prompt in prompt_keys:
                    formatted_prompt = get_formatted_prompt(
                        model_name,
                        prompt,
                        set1=input_line["set1"],
                        set2=input_line["set2"],
                    )

                    print(formatted_prompt)

                    output = pipeline(
                        formatted_prompt, max_new_tokens=128, num_return_sequences=1
                    )

                    print(f"Model: {model_name}, Output: {output}")

                    record = {
                        "model_name": model_name,
                        "split_name": split_name,
                        "output": output,
                    }

                    with open("results.jsonl", "a", encoding="utf-8") as f:
                        f.write(json.dumps(record) + "\n")

                break  # For testing, remove this to process all examples


if __name__ == "__main__":
    # Parse arguments for settings file path
    if len(sys.argv) > 1:
        settings_file_path = sys.argv[1]

        init_settings(settings_file_path)
    else:
        print("No settings file path provided.")
        # Quit
        sys.exit(1)

    run_metric()
