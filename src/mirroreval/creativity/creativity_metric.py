from mirroreval.call_hf_model import call_hf_model


def run_metric():
    input_line = {
        "src": "ingredient pan fry add",
        "set1": [
            "Add the ingredient to the pan and fry it.",
            "Fry the ingredient in the pan after you add it.",
            "Once you add the ingredient to the pan, begin to fry.",
            "To cook, add the chosen ingredient into the pan and start to fry.",
        ],
        "set2": [
            "Fry the ingredient in the pan after you add it.",
            "After being added to the pan, the ingredient is fried.",
            "Once you add the ingredient to the pan, begin to fry.",
            "Begin frying once the ingredient is added to the pan.",
        ],
        "set1_label": "original",
        "set2_label": "para_b",
        "Quality_Set1": 4,
        "Quality_Set2": 4,
        "Diversity_Set1": 3,
        "Diversity_Set2": 2,
        "llm_quality": 2,
        "llm_diversity": 0,
    }

    input_text = f"Rate the diversity and quality of each of the next two sentences: {input_line["set1"]}, {input_line["set2"]}. Provide your answer as a tuple (Diversity_Set1, Quality_Set1, Diversity_Set2, Quality_Set2) where Diversity and Quality are rated on a scale from 1 to 4."

    print(input_text)

    model_name_1 = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    model_name_2 = "google/gemma-7b"

    model_name_3 = "Qwen/Qwen3-0.6B"

    output_1 = call_hf_model(model_name_1, input_text)

    output_2 = call_hf_model(model_name_2, input_text)

    output_3 = call_hf_model(model_name_3, input_text)

    print(f"Model: {model_name_1}, Output: {output_1}")
    print(f"Model: {model_name_2}, Output: {output_2}")
    print(f"Model: {model_name_3}, Output: {output_3}")


if __name__ == "__main__":
    run_metric()
