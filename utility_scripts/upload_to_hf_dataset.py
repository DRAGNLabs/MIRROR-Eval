import json
import os
from datasets import Dataset, DatasetDict, load_dataset, Features, Value, Sequence
from huggingface_hub import HfApi, HfFolder

# === USER CONFIGURATION ===
DATA_DIR = "./data"  # path to directory with your .json files
REPO_ID = "royal42/gcr-diversity"  # Hugging Face dataset repo
SPLIT_BY_FILENAME = True  # whether to create train/test splits based on filenames


# Define the canonical schema
FEATURES = Features(
    {
        "src": Value("string"),
        "set1": Sequence(Value("string")),
        "set2": Sequence(Value("string")),
        "set1_label": Value("string"),
        "set2_label": Value("string"),
        "Quality_Set1": Value("float64"),
        "Quality_Set2": Value("float64"),
        "Diversity_Set1": Value("float64"),
        "Diversity_Set2": Value("float64"),
        "llm_quality": Value("float64"),
        "llm_diversity": Value("float64"),
    }
)


def load_json_files(data_dir):
    """Load all JSON files and combine their entries."""
    all_data = {}
    for fname in os.listdir(data_dir):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(data_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]
        split_name = os.path.splitext(fname)[0] if SPLIT_BY_FILENAME else "train"
        all_data[split_name] = data
        print(f"Loaded {len(data)} samples from {fname}")
    return all_data


def build_dataset(all_data):
    """Create a HuggingFace DatasetDict."""
    ds_dict = {}
    for split, items in all_data.items():
        ds = Dataset.from_list(items)
        ds = ds.cast(FEATURES)
        ds_dict[split] = ds
    return DatasetDict(ds_dict)


def push_to_hub(ds_dict, repo_id):
    """Upload dataset to the Hugging Face Hub."""
    print(f"Pushing dataset to {repo_id}...")
    ds_dict.push_to_hub(repo_id)
    print("✅ Successfully uploaded!")


def main():
    all_data = load_json_files(DATA_DIR)
    ds_dict = build_dataset(all_data)
    print(ds_dict)
    push_to_hub(ds_dict, REPO_ID)


if __name__ == "__main__":
    main()
