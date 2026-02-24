from abc import ABC, abstractmethod
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_distances
from typing import Dict, Type, List, Any, Iterator

from mirroreval.config import settings
from mirroreval.creativity_development.prompts import (
    get_prompt_names,
    get_formatted_prompt,
)
from mirroreval.hf_utilities import get_hf_pipeline, get_hf_model, get_hf_tokenizer
from mirroreval.logger import logger

BATCH_SIZE = 128  # Adjust based on your model/GPU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

METRICS: Dict[str, Type["MetricInterface"]] = {}


def register_metric(name: str):
    """Decorator to register a metric function by name."""

    def decorator(cls: Type["MetricInterface"]):
        METRICS[name] = cls
        return cls

    return decorator


class MetricInterface(ABC):
    """Abstract base class for metrics."""

    @abstractmethod
    def __call__(self, dataset) -> Iterator[Any]:
        pass


@register_metric("llm-as-a-judge")
class LLMAsAJudge(MetricInterface):

    def chunked(self, iterable, n):
        """Yield successive n-sized chunks from iterable."""
        for i in range(0, len(iterable), n):
            yield iterable[i : i + n]

    def __call__(self, dataset) -> Iterator[dict[str, Any]]:
        """
        Takes a dataset generator and yields a dictionary with the following keys:
        - input_id: The index of the input line in the dataset
        - split: The dataset split (e.g., train, validation, test)
        - metric: The name of the metric ("llm-as-a-judge")
        - model_name: The name of the model used
        - prompt: The prompt name used
        - output: The raw output from the model
        - score: The score assigned by the model (0 for set1 better, 1
                    for set2 better, -1 for tie)
        - accuracy: 1 if the model's score matches the ground truth, 0 otherwise
        """
        models = settings.creativity.llm_judge_models

        prompt_names = get_prompt_names()

        for model_name in models:
            pipeline = get_hf_pipeline(model_name)

            sample_chunk = []
            for index, input_line in enumerate(dataset):
                logger.info(f"Processing input line {index} with model {model_name}")
                set1, set2 = (
                    input_line["set1"],
                    input_line["set2"],
                )

                for prompt_name in prompt_names:
                    formatted_prompt = get_formatted_prompt(
                        model_name=model_name,
                        prompt_name=prompt_name,
                        prompt_type="multiple",
                        set1=set1,
                        set2=set2,
                    )

                    sample_chunk.append(
                        (
                            index,
                            input_line,
                            prompt_name,
                            formatted_prompt,
                        )
                    )

                if len(sample_chunk) >= BATCH_SIZE:
                    prompt_chunk = [sample[3] for sample in sample_chunk]
                    outputs = pipeline(
                        prompt_chunk, max_new_tokens=64, num_return_sequences=1
                    )

                    # Calculate scores
                    outputs = [
                        output[0]["generated_text"][-1]["content"] for output in outputs
                    ]
                    scores = []
                    # TODO: what if the output is malformed?
                    for output in outputs:
                        try:
                            set1_score, set2_score = map(
                                int, output.strip("()").split(",")
                            )
                        except Exception as e:
                            print(
                                f"Error parsing output: {output}. Exception: {e}",
                                flush=True,
                            )
                            # Skip this sample
                            set1_score, set2_score = 0, 0
                        if set1_score > set2_score:
                            scores.append(0)
                        elif set2_score > set1_score:
                            scores.append(1)
                        else:
                            scores.append(-1)  # Tie

                    for sample, output, score in zip(sample_chunk, outputs, scores):
                        output_dict = {
                            "input_id": sample[0],
                            "input": sample[1],
                            "metric": "llm-as-a-judge",
                            "model_name": model_name,
                            "prompt": sample[2],
                            "output": output,
                            "score": score,
                        }
                        yield output_dict

                    sample_chunk = []


@register_metric("self-avgcosine")
class SelfAvgCosine(MetricInterface):
    def __init__(self):
        # TODO: load these before potential slurm job
        self.model = get_hf_model("FacebookAI/roberta-large").to(device).eval()
        self.tokenizer = get_hf_tokenizer("FacebookAI/roberta-large")

    def __call__(self, dataset):
        # CREDIT: github.com/LivNLP/Evaluating-Diversity-Metrics

        for index, input_line in enumerate(dataset):
            logger.info(f"Processing input line {index} with SelfAvgCosine metric")
            set1, set2 = input_line["set1"], input_line["set2"]
            set1_score = self.metric(set1)
            set2_score = self.metric(set2)

            if set1_score < set2_score:
                score = 0
            elif set2_score < set1_score:
                score = 1
            else:
                score = -1  # tie

            yield {
                "input_id": index,
                "input": input_line,
                "metric": "self-avgcosine",
                "set1_avgcosine": float(set1_score),
                "set2_avgcosine": float(set2_score),
                "score": score,
            }

    def metric(self, sentences: List[str]) -> float:
        inputs = self.tokenizer(
            sentences, padding=True, truncation=True, return_tensors="pt"
        )
        inputs = {
            key: value.to(device) for key, value in inputs.items()
        }  # Move input data to GPU

        # Get the embeddings
        with torch.no_grad():
            embeddings = self.model(
                **inputs, output_hidden_states=True, return_dict=True
            ).pooler_output

        # Calculate cosine similarities and sum them up for each sentence
        embeddings_norm = embeddings / embeddings.norm(dim=1, keepdim=True)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings_norm, embeddings_norm.T)

        # Exclude self-comparisons
        mask = torch.ones_like(similarity_matrix) - torch.eye(
            len(embeddings), device=embeddings.device
        )
        total_similarities = (similarity_matrix * mask).sum().item()
        num_comparisons = mask.sum().item()

        # Calculate average similarity
        average_similarity = total_similarities / num_comparisons
        return average_similarity


@register_metric("chamfer")
class Chamfer(MetricInterface):
    def __init__(self):
        self.model = get_hf_model("FacebookAI/roberta-large").to(device).eval()
        self.tokenizer = get_hf_tokenizer("FacebookAI/roberta-large")

    def __call__(self, dataset):
        # CREDIT: github.com/LivNLP/Evaluating-Diversity-Metrics
        for index, input_line in enumerate(dataset):
            logger.info(f"Processing input line {index} with Chamfer metric")
            set1, set2 = input_line["set1"], input_line["set2"]
            set1_score = self.metric(set1)
            set2_score = self.metric(set2)

            if set1_score > set2_score:
                score = 0
            elif set2_score > set1_score:
                score = 1
            else:
                score = -1  # tie

            yield {
                "input_id": index,
                "input": input_line,
                "metric": "chamfer",
                "set1_chamfer": float(set1_score),
                "set2_chamfer": float(set2_score),
                "score": score,
            }

    def metric(self, sentences: List[str]) -> float:

        inputs = self.tokenizer(
            sentences, padding=True, truncation=True, return_tensors="pt"
        )
        inputs = {
            key: value.to(device) for key, value in inputs.items()
        }  # Move input data to GPU

        # Get the embeddings
        with torch.no_grad():
            embeddings = self.model(
                **inputs, output_hidden_states=True, return_dict=True
            ).pooler_output

        # Calculate cosine similarities and sum them up for each sentence
        embeddings_norm = embeddings / embeddings.norm(dim=1, keepdim=True)
        embeddings_np = embeddings_norm.cpu().numpy()
        cosine_dist_matrix = cosine_distances(embeddings_np)
        np.fill_diagonal(cosine_dist_matrix, np.inf)
        min_distances = np.min(cosine_dist_matrix, axis=1)
        chamfer_distance_value = np.mean(min_distances)

        return chamfer_distance_value
