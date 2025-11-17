import copy

PROMPT_NAMES = ["scale", "scale_with_examples", "category", "category_with_examples"]

PROMPTS_MULTIPLE = {
    # Detailed source: [Evaluating the Evaluation of Diversity in Commonsense Generation](https://aclanthology.org/2025.acl-long.1181/) (Zhang et al., ACL 2025)
    # Modified to ellict more correct responses. Adds more details on formatting.
    "scale": """
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
    DO NOT output any reasoning or explanations. Output the scores in a tuple, in the following format:

    Example Output:  (Diversity_Score_Set_1, Diversity_Score_Set_2)

    Here are the two sets of sentences to evaluate:
    Set 1: {set1}
    Set 2: {set2}
    """,
    # Detailed with 5-shot examples
    "scale_with_examples": """
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
    DO NOT output any reasoning or explanations. Output the scores in a tuple, in the following format:

    Example Output:  (Diversity_Score_Set_1, Diversity_Score_Set_2)

    Here are some examples of how sentences might be scored:
    Set 1: ["Add the ingredient to the pan and fry it.", "Fry the ingredient in the pan after you add it.", "Once you add the ingredient to the pan, begin to fry.", "To cook, add the chosen ingredient into the pan and start to fry."]
    Set 2: ["Fry the ingredient in the pan after you add it.", "After being added to the pan, the ingredient is fried.", "Once you add the ingredient to the pan, begin to fry.", "Begin frying once the ingredient is added to the pan."]
    Output: (3,2)

    Set 1: ["She holds the kite up to ride on her snowboard as it pulls her swiftly.", "The kite is held by her as she swiftly rides on her snowboard, being pulled along.", "On her snowboard, she is held by the kite as it swiftly pulls her along.", "As the kite pulled forward, she held on tight while riding her snowboard."]
    Set 2: ["While riding his snowboard, he is held by the kite allowing the wind to pull him.", "He holds onto the kite while he rides his snowboard, letting the wind pull him.", "While snowboarding, he used a kite to pull him along, holding tightly as he rode.", "She holds the kite up to ride on her snowboard as it pulls her swiftly."]
    Output: (4,3)

    Set 1: ["Children run to collect eggs in a basket during the Easter egg hunt.", "Every morning, she collects eggs in a basket after a quick run around the farm.", "He ran to collect the fallen eggs and carefully placed them back in the basket.", "During the relay, each team member must run to collect an egg and place it in their team's basket."]
    Set 2: ["During the Easter egg hunt, eggs are collected by children as they run to put them in a basket.", "Children run to collect eggs in a basket during the Easter egg hunt.", "Every morning, she collects eggs in a basket after a quick run around the farm.", "He ran to collect the fallen eggs and carefully placed them back in the basket."]
    Output: (5,3)

    Here are the two sets of sentences to evaluate:
    Set 1: {set1}
    Set 2: {set2}
    """,
    "category": """
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

    Diversity Scoring Guidelines:
    Assign the set with more diversity a score of 1. Assign the other set a score of 0.

    Output:  Based on the above criteria, assign the set with more diversity a score of 1, and the other set a score of 0. ONLY output the scores.
    DO NOT output any reasoning or explanations. Output the scores in a tuple, in the following format:

    Example Output:  (Diversity_Score_Set_1, Diversity_Score_Set_2)

    Here are the two sets of sentences to evaluate:
    Set 1: {set1}
    Set 2: {set2}
    """,
    # Detailed with 5-shot examples
    "category_with_examples": """
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

    Diversity Scoring Guidelines:
    Assign the set with more diversity a score of 1. Assign the other set a score of 0.

    Output:  Based on the above criteria, assign the set with more diversity a score of 1, and the other set a score of 0. ONLY output the scores.
    DO NOT output any reasoning or explanations. Output the scores in a tuple, in the following format:

    Example Output:  (Diversity_Score_Set_1, Diversity_Score_Set_2)

    Here are some examples of how sentences might be scored:
    Set 1: ["Add the ingredient to the pan and fry it.", "Fry the ingredient in the pan after you add it.", "Once you add the ingredient to the pan, begin to fry.", "To cook, add the chosen ingredient into the pan and start to fry."]
    Set 2: ["Fry the ingredient in the pan after you add it.", "After being added to the pan, the ingredient is fried.", "Once you add the ingredient to the pan, begin to fry.", "Begin frying once the ingredient is added to the pan."]
    Output: (1,0)

    Set 1: ["She holds the kite up to ride on her snowboard as it pulls her swiftly.", "The kite is held by her as she swiftly rides on her snowboard, being pulled along.", "On her snowboard, she is held by the kite as it swiftly pulls her along.", "As the kite pulled forward, she held on tight while riding her snowboard."]
    Set 2: ["While riding his snowboard, he is held by the kite allowing the wind to pull him.", "He holds onto the kite while he rides his snowboard, letting the wind pull him.", "While snowboarding, he used a kite to pull him along, holding tightly as he rode.", "She holds the kite up to ride on her snowboard as it pulls her swiftly."]
    Output: (1,0)

    Set 1: ["Children run to collect eggs in a basket during the Easter egg hunt.", "Every morning, she collects eggs in a basket after a quick run around the farm.", "He ran to collect the fallen eggs and carefully placed them back in the basket.", "During the relay, each team member must run to collect an egg and place it in their team's basket."]
    Set 2: ["During the Easter egg hunt, eggs are collected by children as they run to put them in a basket.", "Children run to collect eggs in a basket during the Easter egg hunt.", "Every morning, she collects eggs in a basket after a quick run around the farm.", "He ran to collect the fallen eggs and carefully placed them back in the basket."]
    Output: (0,1)

    Here are the two sets of sentences to evaluate:
    Set 1: {set1}
    Set 2: {set2}
    """,
}

PROMPTS_SINGLE = {
    "scale": """
    Task Description:
    You are presented with a set of sentences. The set contains sentences around a common theme.
    Your task is to evaluate the set based on its adherence to commonsense (quality) and its diversity, focusing particularly on redundancy within the set.
    Subtle differences in reasoning or approach should also be recognized.
    The sentence set should be cohesive around the same theme, and diversity should be considered in terms of exploring different aspects of that theme.

    Diversity Evaluation Criteria:
    1. Low Redundancy: Sentences should exhibit low lexical and semantic similarity.
    2. Degree of Diversity: Set with more paraphrased sentences or repetitive themes have lower diversity.
    3. Comprehensive Diversity: The sentences in the set should enrich the theme without compromising realism and commonsense.

    Diversity Scoring Guidelines:
    5 Points: Sentences explore a wide range of aspects of the theme with negligible redundancy.
    4 Points: Sentences cover different aspects of the theme with minor lexical/semantic overlap.
    3 Points: Sentences have some diversity but noticeable redundancy.
    2 Points: Sentences are mostly repetitive with limited exploration of the theme.
    1 Point: Sentences are highly redundant in with almost no diversity.

    Output:  Based on the above criteria, assign a score for quality and diversity to the set, ranging from 1 to 5 points. ONLY output the score.
    DO NOT output any reasoning or explanations. Output the score in a tuple, in the following format:

    Example Output:  (Diversity_Score)

    Here is the set of sentences to evaluate:
    Set: {set}
    """,
    # Detailed with 5-shot examples
    "scale_with_examples": """
    Task Description:
    You are presented with a set of sentences. The set contains sentences around a common theme.
    Your task is to evaluate the set based on its adherence to commonsense (quality) and its diversity, focusing particularly on redundancy within the set.
    Subtle differences in reasoning or approach should also be recognized.
    The sentence set should be cohesive around the same theme, and diversity should be considered in terms of exploring different aspects of that theme.

    Diversity Evaluation Criteria:
    1. Low Redundancy: Sentences should exhibit low lexical and semantic similarity.
    2. Degree of Diversity: Set with more paraphrased sentences or repetitive themes have lower diversity.
    3. Comprehensive Diversity: The sentences in the set should enrich the theme without compromising realism and commonsense.

    Diversity Scoring Guidelines:
    5 Points: Sentences explore a wide range of aspects of the theme with negligible redundancy.
    4 Points: Sentences cover different aspects of the theme with minor lexical/semantic overlap.
    3 Points: Sentences have some diversity but noticeable redundancy.
    2 Points: Sentences are mostly repetitive with limited exploration of the theme.
    1 Point: Sentences are highly redundant in with almost no diversity.

    Output:  Based on the above criteria, assign a score for quality and diversity to the set, ranging from 1 to 5 points. ONLY output the score.
    DO NOT output any reasoning or explanations. Output the score in a tuple, in the following format:

    Example Output:  (Diversity_Score)

    Here are some examples of how sentences might be scored:
    Set: ["Fry the ingredient in the pan after you add it.", "After being added to the pan, the ingredient is fried.", "Once you add the ingredient to the pan, begin to fry.", "Begin frying once the ingredient is added to the pan."]
    Output: (2)

    Set: ["She holds the kite up to ride on her snowboard as it pulls her swiftly.", "The kite is held by her as she swiftly rides on her snowboard, being pulled along.", "On her snowboard, she is held by the kite as it swiftly pulls her along.", "As the kite pulled forward, she held on tight while riding her snowboard."]
    Output: (4)

    Set: ["During the Easter egg hunt, eggs are collected by children as they run to put them in a basket.", "Children run to collect eggs in a basket during the Easter egg hunt.", "Every morning, she collects eggs in a basket after a quick run around the farm.", "He ran to collect the fallen eggs and carefully placed them back in the basket."]
    Output: (3)

    Here is the set of sentences to evaluate:
    Set: {set}
    """,
    "category": """
    Task Description:
    You are presented with a set of sentences. The set contains sentences around a common theme.
    Your task is to evaluate the set based on its adherence to commonsense (quality) and its diversity, focusing particularly on redundancy within the set.
    Subtle differences in reasoning or approach should also be recognized.
    The sentence set should be cohesive around the same theme, and diversity should be considered in terms of exploring different aspects of that theme.

    Diversity Evaluation Criteria:
    1. Low Redundancy: Sentences should exhibit low lexical and semantic similarity.
    2. Degree of Diversity: Set with more paraphrased sentences or repetitive themes have lower diversity.
    3. Comprehensive Diversity: The sentences in the set should enrich the theme without compromising realism and commonsense.

    Diversity Scoring Guidelines:
    Assign the set a 1 if the sentences are diverse, and a 0 if they are not.

    Output:  Based on the above criteria, assign the set a 1 if the sentences are diverse, and a 0 if they are not. ONLY output the score.
    DO NOT output any reasoning or explanations. Output the score in a tuple, in the following format:

    Example Output:  (Diversity_Score)

    Here is the set of sentences to evaluate:
    Set: {set}
    """,
    # Detailed with 5-shot examples
    "category_with_examples": """
    Task Description:
    You are presented with a set of sentences. The set contains sentences around a common theme.
    Your task is to evaluate the set based on its adherence to commonsense (quality) and its diversity, focusing particularly on redundancy within the set.
    Subtle differences in reasoning or approach should also be recognized.
    The sentence set should be cohesive around the same theme, and diversity should be considered in terms of exploring different aspects of that theme.

    Diversity Evaluation Criteria:
    1. Low Redundancy: Sentences should exhibit low lexical and semantic similarity.
    2. Degree of Diversity: Set with more paraphrased sentences or repetitive themes have lower diversity.
    3. Comprehensive Diversity: The sentences in the set should enrich the theme without compromising realism and commonsense.

    Diversity Scoring Guidelines:
    Assign the set a 1 if the sentences are diverse, and a 0 if they are not.

    Output:  Based on the above criteria, assign the set a 1 if the sentences are diverse, and a 0 if they are not. ONLY output the score.
    DO NOT output any reasoning or explanations. Output the score in a tuple, in the following format:

    Example Output:  (Diversity_Score)

    Here are some examples of how sentences might be scored:
    Set: ["Add the ingredient to the pan and fry it.", "Fry the ingredient in the pan after you add it.", "Once you add the ingredient to the pan, begin to fry.", "To cook, add the chosen ingredient into the pan and start to fry."]
    Output: (1)

    Set: ["While riding his snowboard, he is held by the kite allowing the wind to pull him.", "He holds onto the kite while he rides his snowboard, letting the wind pull him.", "While snowboarding, he used a kite to pull him along, holding tightly as he rode.", "She holds the kite up to ride on her snowboard as it pulls her swiftly."]
    Output: (0)

    Set: ["During the Easter egg hunt, eggs are collected by children as they run to put them in a basket.", "Children run to collect eggs in a basket during the Easter egg hunt.", "Every morning, she collects eggs in a basket after a quick run around the farm.", "He ran to collect the fallen eggs and carefully placed them back in the basket."]
    Output: (1)

    Here is the set of sentences to evaluate:
    Set: {set}
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
}


def get_prompt(prompt_name, prompt_type, **kwargs):
    if prompt_type == "single":
        # If set1 is provided, pass that as set
        if "set1" in kwargs:
            return PROMPTS_SINGLE[prompt_name].format(set=kwargs["set1"])
        elif "set2" in kwargs:
            return PROMPTS_SINGLE[prompt_name].format(set=kwargs["set2"])
    elif prompt_type == "multiple":
        return PROMPTS_MULTIPLE[prompt_name].format(**kwargs)
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")


def get_prompt_names():
    return PROMPT_NAMES


def get_formatted_prompt(model_name, prompt_name, prompt_type, **kwargs):
    # Deep copy to avoid mutating the original
    prompt_template = copy.deepcopy(SYSTEM_PROMPTS[model_name])
    actual_prompt = get_prompt(
        prompt_name=prompt_name, prompt_type=prompt_type, **kwargs
    )
    for message in prompt_template:
        if message["role"] == "user":
            message["content"] = message["content"].format(prompt=actual_prompt)
    return prompt_template
