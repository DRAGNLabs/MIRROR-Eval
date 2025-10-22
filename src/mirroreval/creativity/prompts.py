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

    Example Output:  "Set 1": (Sentence Set 1) "Set 2": (Sentence Set 2) "Diversity_Score_Set1": (score), "Diversity_Score_Set2": (score)

    Here are the two sets of sentences to evaluate:
    Set 1: {set1}
    Set 2: {set2}
    """,
}


def get_prompt(name, **kwargs):
    return PROMPTS[name].format(**kwargs)
