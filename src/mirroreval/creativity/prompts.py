PROMPTS = {
    "default": "Rate the diversity and quality of each of the next two sentences: {set1}, {set2}. Provide your answer as a tuple (Diversity_Set1, Quality_Set1, Diversity_Set2, Quality_Set2) where Diversity and Quality are rated on a scale from 1 to 4.",
}


def get_prompt(name, **kwargs):
    return PROMPTS[name].format(**kwargs)
