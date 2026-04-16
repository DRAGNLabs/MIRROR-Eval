import torch
from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import pipeline, AutoModel, AutoTokenizer, BitsAndBytesConfig

from mirroreval.logger import logger


def has_chat_template(model_name):
    """Check if a model's tokenizer supports chat templates."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer.chat_template is not None


def _try_accelerate():
    """Check if accelerate is available for device_map='auto'."""
    try:
        import accelerate  # noqa: F401
        return True
    except ImportError:
        return False


def call_hf_model(model_name, input_text):
    """Call a Hugging Face model for text generation. Simple for single calls."""
    kwargs = {"model_kwargs": {"dtype": torch.bfloat16}}
    if _try_accelerate():
        kwargs["device_map"] = "auto"
    pipe = pipeline("text-generation", model=model_name, **kwargs)
    return pipe(input_text, max_length=50, num_return_sequences=1)


def get_hf_pipeline(model_name, task="text-generation", quantize=None):
    """Get a Hugging Face pipeline for a given model and task.

    Args:
        model_name: HuggingFace model ID or local path.
        task: Pipeline task type.
        quantize: None for no quantization, "4bit" or "8bit" for bitsandbytes quantization.
    """
    model_kwargs = {}
    if quantize == "4bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quantize == "8bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        model_kwargs["dtype"] = torch.bfloat16

    kwargs = {"model_kwargs": model_kwargs}
    if _try_accelerate():
        kwargs["device_map"] = "auto"
    pipe = pipeline(task, model=model_name, **kwargs)
    return pipe


def download_from_hf(repo_id):
    """Download a model, tokenizer, or other artifact from HuggingFace Hub."""
    cache_path = snapshot_download(repo_id=repo_id)
    logger.info(f"Downloaded {repo_id} to {cache_path}")


def load_hf_dataset(dataset_name):
    dataset = load_dataset(dataset_name)
    return dataset


def get_hf_model(model_name):
    model = AutoModel.from_pretrained(model_name)
    return model


def get_hf_tokenizer(tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return tokenizer
