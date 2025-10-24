PROMPTS = {
    "default": """
    Rate the diversity and quality of each of the next two sentences: {set1}, {set2}.
    Provide your answer as a tuple (Diversity_Set1, Quality_Set1, Diversity_Set2, Quality_Set2) where Diversity and Quality are rated on a scale from 1 to 4.
    """.strip(),
    # Detailed source: [Evaluating the Evaluation of Diversity in Commonsense Generation](https://aclanthology.org/2025.acl-long.1181/) (Zhang et al., ACL 2025)
    "detailed": """
    Task Description:
    You are presented with two sets of sentences, Set 1 and Set 2. Each set contains sentences around a common theme.
    Your task is to evaluate each set based on their adherence to commonsense (quality) and their diversity, focusing particularly on redundancy within the sets.
    Subtle differences in reasoning or approach should also be recognized.
    The sentence sets should be cohesive around the same theme, and diversity should be considered in terms of exploring different aspects of that theme.

    Important Notes:  It is crucial to pay close attention to which sentences are in Set 1 and which are in Set 2 when making your evaluations.
    Do not assume any set is superior by default in quality or diversity. Evaluate each set independently based on its own content.

    Diversity Evaluation Criteria:
    1. Low Redundancy: Sentences should exhibit low lexical and semantic similarity.
    2. Degree of Diversity: Sets with more paraphrased sentences or repetitive themes have lower diversity.
    3. Comprehensive Diversity: The sentences in the sets should enrich the theme without compromising realism and commonsense.

    Diversity Scoring Guidelines (for each set):
    5 Points: Sentences explore a wide range of aspects of the theme with negligible redundancy.
    4 Points: Sentences cover different aspects of the theme with minor lexical/semantic overlap.
    3 Points: Sentences have some diversity but noticeable redundancy.
    2 Points: Sentences are mostly repetitive with limited exploration of the theme.
    1 Point: Sentences are highly redundant in with almost no diversity.

    Output:  Based on the above criteria, assign a separate score for quality and diversity to each set, ranging from 1 to 5 points.
    Examples:  "Set 1": (Sentence Set 1) "Set 2": (Sentence Set 2) "Diversity_Score_Set1": (score), "Diversity_Score_Set2": (score)

    {set1}
    {set2}
    """,
    # Modified to ellict more correct responses. Adds more details on formatting.
    "detailed_modified": """
    Task Description:
    You are presented with two sets of sentences, Set 1 and Set 2. Each set contains sentences around a common theme.
    Your task is to evaluate each set based on their adherence to commonsense (quality) and their diversity, focusing particularly on redundancy within the sets.
    Subtle differences in reasoning or approach should also be recognized.
    The sentence sets should be cohesive around the same theme, and diversity should be considered in terms of exploring different aspects of that theme.

    Important Notes:  It is crucial to pay close attention to which sentences are in Set 1 and which are in Set 2 when making your evaluations.
    Do not assume any set is superior by default in quality or diversity. Evaluate each set independently based on its own content.

    Diversity Evaluation Criteria:
    1. Low Redundancy: Sentences should exhibit low lexical and semantic similarity.
    2. Degree of Diversity: Sets with more paraphrased sentences or repetitive themes have lower diversity.
    3. Comprehensive Diversity: The sentences in the sets should enrich the theme without compromising realism and commonsense.

    Diversity Scoring Guidelines (for each set):
    5 Points: Sentences explore a wide range of aspects of the theme with negligible redundancy.
    4 Points: Sentences cover different aspects of the theme with minor lexical/semantic overlap.
    3 Points: Sentences have some diversity but noticeable redundancy.
    2 Points: Sentences are mostly repetitive with limited exploration of the theme.
    1 Point: Sentences are highly redundant in with almost no diversity.

    Output:  Based on the above criteria, assign a separate score for quality and diversity to each set, ranging from 1 to 5 points. ONLY output the scores.
    DO NOT output any reasoning or explanations. Output the scores in the following format:

    Example Output:  "Diversity_Score_Set1": (score), "Diversity_Score_Set2": (score)

    Here are the two sets of sentences to evaluate:
    Set 1: {set1}
    Set 2: {set2}
    """,
    "detailed_with_example": """
    Task Description:
    You are presented with two sets of sentences, Set 1 and Set 2. Each set contains sentences around a common theme.
    Your task is to evaluate each set based on their adherence to commonsense (quality) and their diversity, focusing particularly on redundancy within the sets.
    Subtle differences in reasoning or approach should also be recognized.
    The sentence sets should be cohesive around the same theme, and diversity should be considered in terms of exploring different aspects of that theme.

    Important Notes:  It is crucial to pay close attention to which sentences are in Set 1 and which are in Set 2 when making your evaluations.
    Do not assume any set is superior by default in quality or diversity. Evaluate each set independently based on its own content.

    Diversity Evaluation Criteria:
    1. Low Redundancy: Sentences should exhibit low lexical and semantic similarity.
    2. Degree of Diversity: Sets with more paraphrased sentences or repetitive themes have lower diversity.
    3. Comprehensive Diversity: The sentences in the sets should enrich the theme without compromising realism and commonsense.

    Diversity Scoring Guidelines (for each set):
    5 Points: Sentences explore a wide range of aspects of the theme with negligible redundancy.
    4 Points: Sentences cover different aspects of the theme with minor lexical/semantic overlap.
    3 Points: Sentences have some diversity but noticeable redundancy.
    2 Points: Sentences are mostly repetitive with limited exploration of the theme.
    1 Point: Sentences are highly redundant in with almost no diversity.

    Output:  Based on the above criteria, assign a separate score for quality and diversity to each set, ranging from 1 to 5 points. ONLY output the scores.
    DO NOT output any reasoning or explanations. Output the scores in the following format:

    Example Output:  "Diversity_Score_Set1": (score), "Diversity_Score_Set2": (score)

    Here is an example of how sentences might be scored:
    Set 1: ["Add the ingredient to the pan and fry it.", "Fry the ingredient in the pan after you add it.", "Once you add the ingredient to the pan, begin to fry.", "To cook, add the chosen ingredient into the pan and start to fry."]
    Set 2: ["Fry the ingredient in the pan after you add it.", "After being added to the pan, the ingredient is fried.", "Once you add the ingredient to the pan, begin to fry.", "Begin frying once the ingredient is added to the pan."]
    Diversity_Score_Set1: 2
    Diversity_Score_Set2: 0

    Here are the two sets of sentences to evaluate:
    Set 1: {set1}
    Set 2: {set2}
    """,
    "detailed_with_examples": """
    Task Description:
    You are presented with two sets of sentences, Set 1 and Set 2. Each set contains sentences around a common theme.
    Your task is to evaluate each set based on their adherence to commonsense (quality) and their diversity, focusing particularly on redundancy within the sets.
    Subtle differences in reasoning or approach should also be recognized.
    The sentence sets should be cohesive around the same theme, and diversity should be considered in terms of exploring different aspects of that theme.

    Important Notes:  It is crucial to pay close attention to which sentences are in Set 1 and which are in Set 2 when making your evaluations.
    Do not assume any set is superior by default in quality or diversity. Evaluate each set independently based on its own content.

    Diversity Evaluation Criteria:
    1. Low Redundancy: Sentences should exhibit low lexical and semantic similarity.
    2. Degree of Diversity: Sets with more paraphrased sentences or repetitive themes have lower diversity.
    3. Comprehensive Diversity: The sentences in the sets should enrich the theme without compromising realism and commonsense.

    Diversity Scoring Guidelines (for each set):
    5 Points: Sentences explore a wide range of aspects of the theme with negligible redundancy.
    4 Points: Sentences cover different aspects of the theme with minor lexical/semantic overlap.
    3 Points: Sentences have some diversity but noticeable redundancy.
    2 Points: Sentences are mostly repetitive with limited exploration of the theme.
    1 Point: Sentences are highly redundant in with almost no diversity.

    Output:  Based on the above criteria, assign a separate score for quality and diversity to each set, ranging from 1 to 5 points. ONLY output the scores.
    DO NOT output any reasoning or explanations. Output the scores in the following format:

    Example Output:  "Diversity_Score_Set1": (score), "Diversity_Score_Set2": (score)

    Here are some examples of how sentences might be scored:
    Set 1: ["Add the ingredient to the pan and fry it.", "Fry the ingredient in the pan after you add it.", "Once you add the ingredient to the pan, begin to fry.", "To cook, add the chosen ingredient into the pan and start to fry."]
    Set 2: ["Fry the ingredient in the pan after you add it.", "After being added to the pan, the ingredient is fried.", "Once you add the ingredient to the pan, begin to fry.", "Begin frying once the ingredient is added to the pan."]
    Diversity_Score_Set1: 3
    Diversity_Score_Set2: 2

    Set 1: ["She holds the kite up to ride on her snowboard as it pulls her swiftly.", "The kite is held by her as she swiftly rides on her snowboard, being pulled along.", "On her snowboard, she is held by the kite as it swiftly pulls her along.", "As the kite pulled forward, she held on tight while riding her snowboard."]
    Set 2: ["While riding his snowboard, he is held by the kite allowing the wind to pull him.", "He holds onto the kite while he rides his snowboard, letting the wind pull him.", "While snowboarding, he used a kite to pull him along, holding tightly as he rode.", "She holds the kite up to ride on her snowboard as it pulls her swiftly."]
    Diversity_Score_Set1: 3
    Diversity_Score_Set2: 4

    Set 1: ["Children run to collect eggs in a basket during the Easter egg hunt.", "Every morning, she collects eggs in a basket after a quick run around the farm.", "He ran to collect the fallen eggs and carefully placed them back in the basket.", "During the relay, each team member must run to collect an egg and place it in their team's basket."]
    Set 2: ["During the Easter egg hunt, eggs are collected by children as they run to put them in a basket.", "Children run to collect eggs in a basket during the Easter egg hunt.", "Every morning, she collects eggs in a basket after a quick run around the farm.", "He ran to collect the fallen eggs and carefully placed them back in the basket."]
    Diversity_Score_Set1: 5
    Diversity_Score_Set2: 3

    Here are the two sets of sentences to evaluate:
    Set 1: {set1}
    Set 2: {set2}
    """,
}

SYSTEM_PROMPT_ROLES = {
    "meta-llama/Llama-3.3-70B-Instruct": "system",
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "system",
    "google/gemma-7b-it": "assistant",
    "Qwen/Qwen3-30B-A3B-Instruct-2507": "system",
}


def get_prompt(name, **kwargs):
    return PROMPTS[name].format(**kwargs)


def get_prompt_names():
    return PROMPTS.keys()


def get_system_prompt_role(model_name):
    return SYSTEM_PROMPT_ROLES.get(model_name, "system")
