import json
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import binomtest

results = "./test_results.jsonl"
results_with_scores = "./results_with_scores.jsonl"


def compute_scores(data_path):
    # parse through each line of json
    with open(data_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    num_uh_oh = 0

    for line in lines:
        line = json.loads(line)
        labels = (line["Diversity_Set1"], line["Diversity_Set2"])

        # Get the index of which set in labels has higher diversity
        if labels[0] > labels[1]:
            correct_label = 0
        elif labels[1] > labels[0]:
            correct_label = 1
        else:
            print("UH OH")
            correct_label = -1  # tie
            num_uh_oh += 1

        # First process the score for both sets. This comes in the form (x, y) and we just want to see if x > y
        llm_scores = line["output_both_sets"].split(",")
        print(llm_scores)
        # Remove parentheses
        llm_scores[0] = llm_scores[0].lstrip("(")
        llm_scores[1] = llm_scores[1].rstrip(")")
        if len(llm_scores) != 2:
            continue  # skip invalid lines
        llm_scores = (float(llm_scores[0]), float(llm_scores[1]))
        if llm_scores[0] > llm_scores[1]:
            predicted_label = 0
        elif llm_scores[1] > llm_scores[0]:
            predicted_label = 1
        else:
            predicted_label = -1  # tie

        if predicted_label == correct_label:
            accuracy = 1
        else:
            accuracy = 0

        # Save to results jsonl
        line["predicted_label_both"] = predicted_label
        line["correct_label"] = correct_label
        line["accuracy_both"] = accuracy

        # Process the outputs for the individual sets. They are in the form (1)
        # Strip parantheses and convert to float
        llm_score_set1 = float(line["output_set_1"].strip("()"))
        llm_score_set2 = float(line["output_set_2"].strip("()"))
        if llm_score_set1 > llm_score_set2:
            predicted_label_set = 0
        elif llm_score_set2 > llm_score_set1:
            predicted_label_set = 1
        else:
            predicted_label_set = -1  # tie

        if predicted_label_set == correct_label:
            accuracy_individual_set = 1
        else:
            accuracy_individual_set = 0

        line["predicted_label_individual"] = predicted_label_set
        line["accuracy_individual_set"] = accuracy_individual_set

        # Save to results jsonl with all the other fields from the line
        with open(results_with_scores, "a", encoding="utf-8") as f:
            f.write(json.dumps(line) + "\n")

    print("num_uh_oh:", num_uh_oh)


def compute_accuracy(data_path):

    prompt_types = [
        "scale",
        "scale_with_examples",
        "category",
        "category_with_examples",
    ]
    # Create a score dictionary, containing seperate score count for each prompt type
    prompt_type_scores = {
        pt: {"total": 0, "correct_both": 0, "correct_individual": 0}
        for pt in prompt_types
    }

    with open(data_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = json.loads(line)
        prompt_type = line.get("prompt", "unknown")
        if prompt_type not in prompt_types:
            continue  # skip unknown prompt types
        # Update scores for this prompt type
        prompt_type_scores[prompt_type]["total"] += 1
        prompt_type_scores[prompt_type]["correct_both"] += line.get("accuracy_both", 0)
        prompt_type_scores[prompt_type]["correct_individual"] += line.get(
            "accuracy_individual_set", 0
        )

    # Compute accuracy
    for pt in prompt_types:
        total = prompt_type_scores[pt]["total"]
        correct_both = prompt_type_scores[pt]["correct_both"]
        correct_individual = prompt_type_scores[pt]["correct_individual"]

        accuracy_both = correct_both / total if total > 0 else 0
        accuracy_individual = correct_individual / total if total > 0 else 0

        # Compute 95% confidence intervals using Wilson score interval
        both_ci_low, both_ci_high = proportion_confint(
            accuracy_both, total, alpha=0.05, method="wilson"
        )
        individual_ci_low, individual_ci_high = proportion_confint(
            accuracy_individual, total, alpha=0.05, method="wilson"
        )

        # Do binomial signicance test
        chance_level = 0.5
        p_value_both = binomtest(
            correct_both, total, chance_level, alternative="greater"
        )
        p_value_individual = binomtest(
            correct_individual, total, chance_level, alternative="greater"
        )

        print(f"Prompt type: {pt}")
        print(f"Total examples: {total}")
        print(f"Accuracy (both sets): {accuracy_both:.4f}")
        print(f"Accuracy (individual sets): {accuracy_individual:.4f}")
        print(f"95% CI (both sets): ({both_ci_low:.4f}, {both_ci_high:.4f})")
        print(
            f"95% CI (individual sets): ({individual_ci_low:.4f}, {individual_ci_high:.4f})"
        )
        print(f"P-value (both sets): {p_value_both}")
        print(f"P-value (individual sets): {p_value_individual}")


compute_scores(results)
compute_accuracy(results_with_scores)
