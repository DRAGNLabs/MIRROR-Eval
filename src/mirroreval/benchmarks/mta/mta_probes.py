import torch
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


def train_probes(
    model, tokenizer, probe_df: pd.DataFrame, shuffle_labels: bool = False
):
    """Train per-layer linear probes to detect whether a fact is present.

    Args:
        model: A HuggingFace causal LM loaded with output_hidden_states=True.
        tokenizer: The corresponding tokenizer.
        probe_df: DataFrame with 'prompt_with_fact' and 'prompt_without_fact' columns,
                  pre-filtered to a single fact_id.
        shuffle_labels: If True, randomly permute labels before fitting. Used for
                  the N2 null-control; expected to yield chance-level test accuracy.

    Returns:
        (probes, probe_scores) where probes is dict[layer_idx, LogisticRegression]
        and probe_scores is dict[layer_idx, float] (test accuracy).
    """

    @torch.no_grad()
    def get_last_token_activations(prompt: str) -> dict[int, np.ndarray]:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model(**inputs)

        # hidden_states is a tuple of length (num_layers + 1), each tensor
        # shaped [batch=1, seq_len, hidden_size]. Index 0 is the embedding
        # layer output; indices 1..N are the transformer block outputs.
        return {
            layer_idx: hs[0, -1, :].float().cpu().numpy()
            for layer_idx, hs in enumerate(outputs.hidden_states)
        }

    with_fact = [(p, True) for p in probe_df["prompt_with_fact"]]
    without_fact = [(p, False) for p in probe_df["prompt_without_fact"]]
    pairs = with_fact + without_fact

    rng = np.random.default_rng(seed=42)
    rng.shuffle(pairs)

    prompts = [p for p, _ in pairs]
    labels = np.array([int(has_fact) for _, has_fact in pairs])

    if shuffle_labels:
        # Break the (activation, label) association for the N2 null-control.
        # Uses a separate rng so the train/test split indices stay stable.
        label_rng = np.random.default_rng(seed=1337)
        label_rng.shuffle(labels)

    # Train/test split within probe data (separate from the dataset-level split)
    train_size = int(0.8 * len(pairs))
    prompts_train = prompts[:train_size]
    labels_train = labels[:train_size]
    prompts_test = prompts[train_size:]
    labels_test = labels[train_size:]
    print(
        f"train n={len(prompts_train)} (pos rate {labels_train.mean():.2f}), "
        f"test n={len(prompts_test)} (pos rate {labels_test.mean():.2f})"
    )

    per_layer_activations: dict[int, list[np.ndarray]] = {}
    for prompt in prompts_train:
        activations = get_last_token_activations(prompt)
        for layer_idx, act in activations.items():
            per_layer_activations.setdefault(layer_idx, []).append(act)

    per_layer_test_activations: dict[int, list[np.ndarray]] = {}
    for prompt in prompts_test:
        activations = get_last_token_activations(prompt)
        for layer_idx, act in activations.items():
            per_layer_test_activations.setdefault(layer_idx, []).append(act)

    probes = {}
    probe_scores = {}
    for layer_idx, activations in per_layer_activations.items():
        X_train = np.stack(activations)
        probe = LogisticRegression(max_iter=1000, C=1.0).fit(X_train, labels_train)
        probes[layer_idx] = probe

        X_test = np.stack(per_layer_test_activations[layer_idx])
        test_score = probe.score(X_test, labels_test)
        print(f"Layer {layer_idx}: probe test accuracy = {test_score:.4f}")
        probe_scores[layer_idx] = test_score

    return probes, probe_scores


def apply_probes(probes, activations):
    """Run trained probes on a single set of activations.

    Args:
        probes: dict[layer_idx, LogisticRegression] from train_probes.
        activations: dict[layer_idx, np.ndarray] — one activation vector per layer.

    Returns:
        dict[layer_idx, prediction] where prediction is the probe's binary output.
    """
    predictions = {}
    for layer_idx, probe in probes.items():
        act = activations[layer_idx]
        pred = probe.predict(act.reshape(1, -1))[0]
        predictions[layer_idx] = pred
    return predictions
