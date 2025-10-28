import json
import sys
from collections import defaultdict

from mirroreval.config import init_settings, settings
from mirroreval.creativity.prompts import (
    get_prompt_names,
    get_formatted_prompt,
)
from mirroreval.hf_utilities import get_hf_pipeline, load_hf_dataset


def chunked(iterable, n):
    """Yield successive n-sized chunks from iterable."""
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]


BATCH_SIZE = 32  # Adjust based on your model/GPU


def run_metric():
    dataset = load_hf_dataset("royal42/gcr-diversity")

    models = settings.creativity.models

    prompt_names = get_prompt_names()

    for model_name in models:
        pipeline = get_hf_pipeline(model_name)
        for split_name, split_dataset in dataset.items():
            print(f"--- Split: {split_name} ---")
            print(f"Number of examples: {len(split_dataset)}")
            prompts = []
            meta = []
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

                    prompts.append(formatted_prompt)
                    meta.append((input_line, prompt_name, "both"))

                    ### Generate only with set 1
                    formatted_prompt = get_formatted_prompt(
                        model_name=model_name,
                        prompt_name=prompt_name,
                        prompt_type="single",
                        set1=set1,
                    )

                    prompts.append(formatted_prompt)
                    meta.append((input_line, prompt_name, "set1"))

                    ### Generate only with set 2
                    formatted_prompt = get_formatted_prompt(
                        model_name=model_name,
                        prompt_name=prompt_name,
                        prompt_type="single",
                        set2=set2,
                    )

                    prompts.append(formatted_prompt)
                    meta.append((input_line, prompt_name, "set2"))

                break  # For testing, remove this to process all examples

            # Process in batches
            all_outputs = []
            for prompt_chunk in chunked(prompts, BATCH_SIZE):
                outputs = pipeline(
                    prompt_chunk, max_new_tokens=64, num_return_sequences=1
                )
                for out in outputs:
                    all_outputs.append(out[0]["generated_text"][-1]["content"])

            # Group outputs by input_line and prompt_name
            grouped = defaultdict(dict)
            for (input_line, prompt_name, which), output in zip(meta, all_outputs):
                key = (input_line["src"], prompt_name)
                grouped[key][which] = output

            # Save results
            for (src, prompt_name), outdict in grouped.items():
                input_line = next(x for x in split_dataset if x["src"] == src)
                record = {
                    "model_name": model_name,
                    "split_name": split_name,
                    "output_both_sets": outdict.get("both", ""),
                    "output_set_1": outdict.get("set1", ""),
                    "output_set_2": outdict.get("set2", ""),
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
