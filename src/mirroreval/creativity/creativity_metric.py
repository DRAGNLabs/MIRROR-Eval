import json
import sys

from mirroreval.config import init_settings, settings
from mirroreval.creativity.prompts import (
    get_prompt_names,
    get_formatted_prompt,
)
from mirroreval.hf_utilities import get_hf_pipeline, load_hf_dataset


def run_metric():
    dataset = load_hf_dataset("royal42/gcr-diversity")

    models = settings.creativity.models

    prompt_names = get_prompt_names()

    for model_name in models:
        pipeline = get_hf_pipeline(model_name)
        for split_name, split_dataset in dataset.items():
            print(f"--- Split: {split_name} ---")
            print(f"Number of examples: {len(split_dataset)}")
            # Iterate through dataset
            for input_line in split_dataset:
                set1 = input_line["set1"]
                set2 = input_line["set2"]
                for prompt_name in prompt_names:
                    ### Generate with both sets
                    formatted_prompt = get_formatted_prompt(
                        model_name=model_name,
                        prompt_name=prompt_name,
                        prompt_type="multiple",
                        set1=set1,
                        set2=set2,
                    )

                    output_both_sets = pipeline(
                        formatted_prompt, max_new_tokens=64, num_return_sequences=1
                    )

                    print(f"Model: {model_name}, both sets, Output: {output_both_sets}")

                    ### Generate only with set 1
                    formatted_prompt = get_formatted_prompt(
                        model_name=model_name,
                        prompt_name=prompt_name,
                        prompt_type="single",
                        set1=set1,
                    )

                    output_set_1 = pipeline(
                        formatted_prompt, max_new_tokens=64, num_return_sequences=1
                    )

                    print(f"Model: {model_name}, only set1, Output: {output_set_1}")

                    ### Generate only with set 2

                    formatted_prompt = get_formatted_prompt(
                        model_name=model_name,
                        prompt_name=prompt_name,
                        prompt_type="single",
                        set2=set2,
                    )

                    output_set_2 = pipeline(
                        formatted_prompt, max_new_tokens=64, num_return_sequences=1
                    )

                    print(f"Model: {model_name}, only set2, Output: {output_set_2}")

                    ### Save results

                    record = {
                        "model_name": model_name,
                        "split_name": split_name,
                        "output_both_sets": output_both_sets,
                        "output_set_1": output_set_1,
                        "output_set_2": output_set_2,
                        "prompt": prompt_name,
                        "src": input_line["src"],
                        "set1": input_line["set1"],
                        "set2": input_line["set2"],
                        "set1_label": input_line["set1_label"],
                        "set2_label": input_line["set2_label"],
                        "Quality_Set1": input_line["Quality_Set1"],
                        "Quality_Set2": input_line["Quality_Set2"],
                        "Diversity_Set1": input_line["Diversity_Set1"],
                        "Diversity_Set2": input_line["Diversity_Set2"],
                        "llm_quality": input_line["llm_quality"],
                        "llm_diversity": input_line["llm_diversity"],
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
