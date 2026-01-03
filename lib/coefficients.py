"""Task vector coefficient computation methods.

These methods compute per-layer scaling factors for task vector application.
The goal is to modulate how much each layer is affected by the reasoning amplification.
"""

import re
import numpy as np
import torch
from typing import Optional
from pathlib import Path
from tqdm import tqdm


def uniform_coefficients(num_layers: int, **kwargs) -> dict[str, float]:
    """All layers get coefficient 1.0 (standard uniform application).

    Args:
        num_layers: Total number of layers in the model

    Returns:
        Dictionary mapping layer_i -> 1.0
    """
    return {f"layer_{i}": 1.0 for i in range(num_layers)}


def freeze_last_n_coefficients(num_layers: int, n: int = 5, **kwargs) -> dict[str, float]:
    """Last n layers get coefficient 0.0, rest get 1.0.

    This preserves the output distribution by not modifying the final layers
    that are most sensitive to perturbation.

    Args:
        num_layers: Total number of layers
        n: Number of layers to freeze (from the end)

    Returns:
        Dictionary mapping layer_i -> coefficient
    """
    return {
        f"layer_{i}": 0.0 if i >= num_layers - n else 1.0
        for i in range(num_layers)
    }


def freeze_last_half_coefficients(num_layers: int, **kwargs) -> dict[str, float]:
    """Last half of layers get coefficient 0.0.

    More aggressive freezing that only modifies early layers.

    Args:
        num_layers: Total number of layers

    Returns:
        Dictionary mapping layer_i -> coefficient
    """
    return freeze_last_n_coefficients(num_layers, n=num_layers // 2)


def linear_decay_coefficients(num_layers: int, **kwargs) -> dict[str, float]:
    """Linearly decay from 1.0 (first layer) to 0.0 (last layer).

    Smooth transition that applies more to early layers.

    Args:
        num_layers: Total number of layers

    Returns:
        Dictionary mapping layer_i -> coefficient (1.0 at layer 0, 0.0 at last)
    """
    if num_layers <= 1:
        return {"layer_0": 1.0}
    return {
        f"layer_{i}": 1.0 - (i / (num_layers - 1))
        for i in range(num_layers)
    }


def cosine_decay_coefficients(
    num_layers: int,
    start_fraction: float = 0.6,
    **kwargs
) -> dict[str, float]:
    """Cosine decay starting at a fraction of total layers.

    Applies full coefficient until start_fraction, then smoothly decays
    using a cosine curve.

    Args:
        num_layers: Total number of layers
        start_fraction: Fraction of layers before decay begins (0.0-1.0)

    Returns:
        Dictionary mapping layer_i -> coefficient
    """
    coeffs = {}
    l_start = int(num_layers * start_fraction)

    for i in range(num_layers):
        if i < l_start:
            coeffs[f"layer_{i}"] = 1.0
        else:
            progress = (i - l_start) / max(1, num_layers - l_start - 1)
            coeffs[f"layer_{i}"] = 0.5 * (1 + np.cos(np.pi * progress))

    return coeffs


def fisher_weighted_coefficients(
    num_layers: int,
    fisher_traces: dict[str, float],
    epsilon: float = 1e-8,
    **kwargs
) -> dict[str, float]:
    """Inverse sqrt of Fisher trace per layer.

    Layers with higher Fisher information (more sensitive to perturbation)
    get lower coefficients to preserve output distribution.

    Formula: lambda_l = 1 / sqrt(tr(F_l) + epsilon)
    Normalized so mean coefficient = 1.0

    Args:
        num_layers: Total number of layers (for compatibility, may not be used)
        fisher_traces: Dictionary mapping layer/param name -> Fisher trace value
        epsilon: Small constant for numerical stability

    Returns:
        Dictionary mapping layer/param name -> normalized coefficient
    """
    if not fisher_traces:
        print("Warning: No Fisher traces provided, using uniform coefficients")
        return uniform_coefficients(num_layers)

    coeffs = {}
    for name, trace in fisher_traces.items():
        coeffs[name] = 1.0 / (np.sqrt(trace) + epsilon)

    # Normalize to mean=1 to preserve overall alpha scale
    mean_coef = np.mean(list(coeffs.values()))
    if mean_coef > 0:
        coeffs = {k: v / mean_coef for k, v in coeffs.items()}

    return coeffs


def map_coefficients_to_params(
    layer_coefficients: dict[str, float],
    param_names: list[str],
    num_layers: int,
) -> dict[str, float]:
    """Map layer-level coefficients to individual parameter names.

    Takes coefficients like {"layer_0": 1.0, "layer_1": 0.8, ...} and maps them
    to actual parameter names like "model.layers.0.self_attn.q_proj.weight".

    Args:
        layer_coefficients: Dictionary mapping "layer_i" -> coefficient
        param_names: List of full parameter names
        num_layers: Total number of layers

    Returns:
        Dictionary mapping parameter name -> coefficient
    """
    param_coeffs = {}

    for name in param_names:
        # Extract layer index from parameter name
        match = re.search(r'layers\.(\d+)\.', name)
        if match:
            layer_idx = int(match.group(1))
            layer_key = f"layer_{layer_idx}"
            param_coeffs[name] = layer_coefficients.get(layer_key, 1.0)
        else:
            # Non-layer parameters (embeddings, lm_head, etc.) get 1.0
            param_coeffs[name] = 1.0

    return param_coeffs


def compute_fisher_traces(
    model,
    tokenizer,
    calibration_texts: list[str],
    num_samples: int = 256,
    batch_size: int = 4,
    max_length: int = 512,
    cache_path: Optional[str] = None,
) -> dict[str, float]:
    """Compute diagonal Fisher information trace per parameter.

    Fisher = E[(grad log p(y|x))^2]
    Approximated by averaging squared gradients over calibration data.

    Args:
        model: The model to compute Fisher for
        tokenizer: Tokenizer for the model
        calibration_texts: List of texts for calibration
        num_samples: Number of samples to use
        batch_size: Batch size for computation
        max_length: Maximum sequence length
        cache_path: Optional path to cache/load Fisher traces

    Returns:
        Dictionary mapping parameter name -> Fisher trace (scalar)
    """
    # Check cache
    if cache_path and Path(cache_path).exists():
        print(f"Loading cached Fisher traces from: {cache_path}")
        return torch.load(cache_path)

    model.eval()

    # Initialize accumulators on CPU to save GPU memory
    fisher_accum = {
        name: torch.zeros_like(param, device='cpu')
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    count = 0
    num_batches = (min(num_samples, len(calibration_texts)) + batch_size - 1) // batch_size

    print(f"Computing Fisher traces over {num_samples} samples...")

    for batch_idx in tqdm(range(num_batches), desc="Fisher computation"):
        if count >= num_samples:
            break

        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(calibration_texts), num_samples)
        batch_texts = calibration_texts[batch_start:batch_end]

        # Tokenize
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(model.device)

        # Forward pass with gradients
        model.zero_grad()

        try:
            outputs = model(**inputs, labels=inputs.input_ids)
            log_likelihood = -outputs.loss
            log_likelihood.backward()

            # Accumulate squared gradients
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None and name in fisher_accum:
                        fisher_accum[name] += (param.grad.data ** 2).cpu()

            count += len(batch_texts)

        except Exception as e:
            print(f"  Warning: Batch {batch_idx} failed: {e}")
            continue

    # Compute traces (sum of diagonal)
    fisher_traces = {}
    for name, fisher in fisher_accum.items():
        trace = (fisher / max(count, 1)).sum().item()
        fisher_traces[name] = trace

    print(f"Computed Fisher traces for {len(fisher_traces)} parameters")

    # Cache if path provided
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(fisher_traces, cache_path)
        print(f"Cached Fisher traces to: {cache_path}")

    return fisher_traces


def aggregate_fisher_by_layer(
    fisher_traces: dict[str, float],
    num_layers: int,
) -> dict[str, float]:
    """Aggregate per-parameter Fisher traces to per-layer traces.

    Args:
        fisher_traces: Dictionary mapping parameter name -> Fisher trace
        num_layers: Total number of layers

    Returns:
        Dictionary mapping "layer_i" -> aggregated Fisher trace
    """
    layer_traces = {f"layer_{i}": 0.0 for i in range(num_layers)}

    for name, trace in fisher_traces.items():
        match = re.search(r'layers\.(\d+)\.', name)
        if match:
            layer_idx = int(match.group(1))
            layer_key = f"layer_{layer_idx}"
            if layer_key in layer_traces:
                layer_traces[layer_key] += trace

    return layer_traces


# Registry of all coefficient methods
COEFFICIENT_METHODS = {
    "uniform": uniform_coefficients,
    "freeze_last_5": lambda n, **kw: freeze_last_n_coefficients(n, 5, **kw),
    "freeze_last_half": freeze_last_half_coefficients,
    "linear_decay": linear_decay_coefficients,
    "cosine_decay": cosine_decay_coefficients,
    "fisher": fisher_weighted_coefficients,
}
