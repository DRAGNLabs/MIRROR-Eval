from transformers import pipeline
from huggingface_hub import snapshot_download
from datasets import load_dataset


def call_hf_model(model_name, input_text):
    pipe = pipeline("text-generation", model=model_name)
    return pipe(input_text, max_length=50, num_return_sequences=1)


def download_hf_model(model_name):
    cache_path = snapshot_download(repo_id=model_name)
    print(f"Model downloaded to {cache_path}")


def download_tokenizer(tokenizer_name):
    cache_path = snapshot_download(repo_id=tokenizer_name)
    print(f"Tokenizer downloaded to {cache_path}")


def download_dataset(dataset_name):
    dataset = load_dataset(dataset_name)
    print(f"Dataset {dataset_name} loaded with {len(dataset)} splits.")
