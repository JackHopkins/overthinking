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


def compute_directional_fisher(
    task_vector: dict[str, torch.Tensor],
    fisher_diag: dict[str, torch.Tensor],
    num_layers: int,
) -> dict[int, float]:
    """Compute Fisher information along the task vector direction for each layer.

    Unlike trace-based Fisher (which measures average sensitivity), directional
    Fisher measures sensitivity specifically in the direction we're perturbing.

    Formula: F_dir,l = (tau_l^T F_l tau_l) / ||tau_l||^2
    For diagonal Fisher: F_dir,l = sum_i(F_i * tau_i^2) / sum_i(tau_i^2)

    This addresses the finding that task vectors are orthogonal to trace-based
    Fisher structure (cosine similarity ~0.002), making trace-based weighting
    ineffective.

    Args:
        task_vector: Dictionary mapping parameter name -> task vector tensor
        fisher_diag: Dictionary mapping parameter name -> diagonal Fisher tensor
        num_layers: Total number of layers

    Returns:
        Dictionary mapping layer_idx (int) -> directional Fisher value
    """
    from collections import defaultdict

    # Group parameters by layer
    layer_params = defaultdict(list)
    for name in task_vector:
        match = re.search(r'layers\.(\d+)\.', name)
        if match:
            layer_idx = int(match.group(1))
            if name in fisher_diag:
                layer_params[layer_idx].append(name)

    directional_fisher = {}

    for layer_idx in range(num_layers):
        param_names = layer_params.get(layer_idx, [])
        if not param_names:
            directional_fisher[layer_idx] = 1e-8
            continue

        # Concatenate task vector and Fisher for this layer
        tau_flat = torch.cat([task_vector[n].flatten() for n in param_names])
        fisher_flat = torch.cat([fisher_diag[n].flatten() for n in param_names])

        # Compute directional Fisher: sum(F_i * tau_i^2) / sum(tau_i^2)
        tau_sq = tau_flat ** 2
        tau_norm_sq = tau_sq.sum()

        f_dir = (fisher_flat * tau_sq).sum() / (tau_norm_sq + 1e-10)
        directional_fisher[layer_idx] = f_dir.item()

    return directional_fisher


def compute_layer_tau_norms(
    task_vector: dict[str, torch.Tensor],
    num_layers: int,
) -> dict[int, float]:
    """Compute task vector L2 norm for each layer.

    Large ||tau_l|| indicates layers where the reasoning capability is
    concentrated (where R and M differ most).

    Args:
        task_vector: Dictionary mapping parameter name -> task vector tensor
        num_layers: Total number of layers

    Returns:
        Dictionary mapping layer_idx (int) -> tau norm
    """
    from collections import defaultdict

    layer_params = defaultdict(list)
    for name in task_vector:
        match = re.search(r'layers\.(\d+)\.', name)
        if match:
            layer_idx = int(match.group(1))
            layer_params[layer_idx].append(name)

    tau_norms = {}

    for layer_idx in range(num_layers):
        param_names = layer_params.get(layer_idx, [])
        if not param_names:
            tau_norms[layer_idx] = 0.0
            continue

        tau_flat = torch.cat([task_vector[n].flatten() for n in param_names])
        tau_norms[layer_idx] = tau_flat.norm().item()

    return tau_norms


def directional_fisher_coefficients(
    num_layers: int,
    directional_fisher: dict[int, float],
    epsilon: float = 1e-8,
    **kwargs
) -> dict[str, float]:
    """Compute coefficients as 1/sqrt(F_dir), normalized to mean=1.

    This method uses directional Fisher (sensitivity along task vector direction)
    rather than trace-based Fisher (average sensitivity across all directions).

    Args:
        num_layers: Total number of layers
        directional_fisher: Dictionary mapping layer_idx -> directional Fisher value
        epsilon: Small constant for numerical stability

    Returns:
        Dictionary mapping "layer_i" -> normalized coefficient
    """
    if not directional_fisher:
        print("Warning: No directional Fisher provided, using uniform coefficients")
        return uniform_coefficients(num_layers)

    coeffs = {}
    for layer_idx in range(num_layers):
        f_dir = directional_fisher.get(layer_idx, epsilon)
        coeffs[f"layer_{layer_idx}"] = 1.0 / (np.sqrt(f_dir) + epsilon)

    # Normalize to mean=1
    mean_coef = np.mean(list(coeffs.values()))
    if mean_coef > 0:
        coeffs = {k: v / mean_coef for k, v in coeffs.items()}

    return coeffs


def task_magnitude_coefficients(
    num_layers: int,
    task_vector: dict[str, torch.Tensor],
    directional_fisher: dict[int, float],
    epsilon: float = 1e-8,
    **kwargs
) -> dict[str, float]:
    """Compute coefficients weighted by task vector magnitude / sqrt(directional Fisher).

    This is the RECOMMENDED method for high-alpha task vector application.
    It achieves ~39% lower perplexity than uniform weighting at alpha=4.0.

    Formula: lambda_l = ||tau_l|| / sqrt(F_dir,l)
    Normalized so mean coefficient = 1.0

    Rationale:
    - ||tau_l||: Amplify layers where reasoning signal is strong
    - 1/sqrt(F_dir,l): Amplify layers where output sensitivity is low
    - The product identifies "safe amplification zones" where we can push
      harder without disrupting base model behavior

    Args:
        num_layers: Total number of layers
        task_vector: Dictionary mapping parameter name -> task vector tensor
        directional_fisher: Dictionary mapping layer_idx -> directional Fisher value
        epsilon: Small constant for numerical stability

    Returns:
        Dictionary mapping "layer_i" -> normalized coefficient

    Example:
        >>> # Compute prerequisites
        >>> fisher_diag = compute_fisher_diagonal(model, tokenizer, calibration_data)
        >>> directional_fisher = compute_directional_fisher(task_vector, fisher_diag, num_layers)
        >>> # Get optimal coefficients
        >>> coeffs = task_magnitude_coefficients(num_layers, task_vector, directional_fisher)
        >>> # Apply task vector
        >>> param_coeffs = map_coefficients_to_params(coeffs, param_names, num_layers)
        >>> apply_task_vector(model, task_vector, alpha=3.0, param_coeffs)
    """
    if not directional_fisher:
        print("Warning: No directional Fisher provided, using uniform coefficients")
        return uniform_coefficients(num_layers)

    # Compute tau norms per layer
    tau_norms = compute_layer_tau_norms(task_vector, num_layers)

    coeffs = {}
    for layer_idx in range(num_layers):
        tau_norm = tau_norms.get(layer_idx, 0.0)
        f_dir = directional_fisher.get(layer_idx, epsilon)

        # lambda ~ ||tau|| / sqrt(F_dir)
        coeffs[f"layer_{layer_idx}"] = tau_norm / (np.sqrt(f_dir) + epsilon)

    # Normalize to mean=1
    mean_coef = np.mean(list(coeffs.values()))
    if mean_coef > 0:
        coeffs = {k: v / mean_coef for k, v in coeffs.items()}

    return coeffs


def compute_optimal_coefficients(
    model,
    tokenizer,
    task_vector: dict[str, torch.Tensor],
    calibration_texts: list[str],
    num_samples: int = 256,
    batch_size: int = 4,
    max_length: int = 512,
    cache_path: Optional[str] = None,
) -> dict[str, float]:
    """Compute task magnitude weighted coefficients (the recommended method).

    This is a convenience function that computes diagonal Fisher, directional
    Fisher, and task magnitude coefficients in one call.

    For high-alpha task vector application (alpha >= 2.5), this method provides
    30-40% lower perplexity compared to uniform weighting.

    Args:
        model: The model to compute coefficients for
        tokenizer: Tokenizer for the model
        task_vector: Dictionary mapping parameter name -> task vector tensor
        calibration_texts: List of texts for Fisher calibration
        num_samples: Number of samples for Fisher computation
        batch_size: Batch size for Fisher computation
        max_length: Maximum sequence length
        cache_path: Optional path to cache directional Fisher values

    Returns:
        Dictionary mapping "layer_i" -> normalized coefficient
    """
    num_layers = model.config.num_hidden_layers

    # Check cache for directional Fisher
    if cache_path and Path(cache_path).exists():
        print(f"Loading cached directional Fisher from: {cache_path}")
        try:
            directional_fisher = torch.load(cache_path, map_location="cpu")
            if directional_fisher and len(directional_fisher) > 0:
                return task_magnitude_coefficients(
                    num_layers, task_vector, directional_fisher
                )
        except Exception as e:
            print(f"  Warning: Cache corrupted, recomputing: {e}")
            Path(cache_path).unlink(missing_ok=True)

    # Step 1: Compute diagonal Fisher
    print("Computing diagonal Fisher information...")
    fisher_diag = {}
    for name, param in model.named_parameters():
        fisher_diag[name] = torch.zeros_like(param, device='cpu')

    model.eval()
    count = 0

    for i, text in enumerate(tqdm(calibration_texts[:num_samples], desc="Fisher computation")):
        try:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length
            ).to(model.device)

            model.zero_grad()
            outputs = model(**inputs, labels=inputs.input_ids)
            outputs.loss.backward()

            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None and name in fisher_diag:
                        fisher_diag[name] += (param.grad ** 2).cpu()

            count += 1
            del inputs, outputs

        except Exception as e:
            continue

    # Average
    for name in fisher_diag:
        fisher_diag[name] /= max(count, 1)

    print(f"Computed Fisher from {count} samples")

    # Step 2: Compute directional Fisher
    print("Computing directional Fisher...")
    directional_fisher = compute_directional_fisher(task_vector, fisher_diag, num_layers)

    # Cache if path provided
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(directional_fisher, cache_path)
        print(f"Cached directional Fisher to: {cache_path}")

    # Step 3: Compute task magnitude coefficients
    return task_magnitude_coefficients(num_layers, task_vector, directional_fisher)


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
    min_successful_batches: int = 2,
) -> dict[str, float]:
    """Compute diagonal Fisher information trace per parameter.

    Fisher = E[(grad log p(y|x))^2]
    Approximated by averaging squared gradients over calibration data.

    Gracefully handles OOM errors by:
    1. Catching per-batch errors and continuing
    2. Cleaning up GPU memory between batches
    3. Returning None if too few batches succeed (caller should fallback)

    Args:
        model: The model to compute Fisher for
        tokenizer: Tokenizer for the model
        calibration_texts: List of texts for calibration
        num_samples: Number of samples to use
        batch_size: Batch size for computation
        max_length: Maximum sequence length
        cache_path: Optional path to cache/load Fisher traces
        min_successful_batches: Minimum batches needed for valid result

    Returns:
        Dictionary mapping parameter name -> Fisher trace (scalar)
        Returns None if computation failed (caller should use fallback)
    """
    import gc

    # Check cache
    if cache_path and Path(cache_path).exists():
        print(f"Loading cached Fisher traces from: {cache_path}")
        try:
            traces = torch.load(cache_path, map_location="cpu", weights_only=True)
            if traces and len(traces) > 0:
                return traces
        except Exception as e:
            print(f"  Warning: Cache corrupted, recomputing: {e}")
            Path(cache_path).unlink(missing_ok=True)

    model.eval()

    # Initialize accumulators on CPU to save GPU memory
    fisher_accum = {
        name: torch.zeros_like(param, device='cpu')
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    count = 0
    successful_batches = 0
    failed_batches = 0
    num_batches = (min(num_samples, len(calibration_texts)) + batch_size - 1) // batch_size

    print(f"Computing Fisher traces over {min(num_samples, len(calibration_texts))} samples (batch_size={batch_size})...")

    for batch_idx in tqdm(range(num_batches), desc="Fisher computation"):
        if count >= num_samples:
            break

        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(calibration_texts), num_samples)
        batch_texts = calibration_texts[batch_start:batch_end]

        try:
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

            outputs = model(**inputs, labels=inputs.input_ids)
            log_likelihood = -outputs.loss
            log_likelihood.backward()

            # Accumulate squared gradients
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None and name in fisher_accum:
                        fisher_accum[name] += (param.grad.data ** 2).cpu()

            count += len(batch_texts)
            successful_batches += 1

            # Clean up to free memory
            del inputs, outputs, log_likelihood
            model.zero_grad(set_to_none=True)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                failed_batches += 1
                # Clean up after OOM
                model.zero_grad(set_to_none=True)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Don't spam the console
                if failed_batches <= 3:
                    print(f"  Warning: Batch {batch_idx} OOM, skipping (GPU memory pressure)")
                elif failed_batches == 4:
                    print(f"  Warning: Multiple OOM errors, suppressing further warnings...")
            else:
                print(f"  Warning: Batch {batch_idx} failed: {e}")
            continue

        except Exception as e:
            print(f"  Warning: Batch {batch_idx} failed: {e}")
            failed_batches += 1
            continue

    # Check if we have enough successful batches
    if successful_batches < min_successful_batches:
        print(f"  Fisher computation failed: only {successful_batches}/{num_batches} batches succeeded")
        print(f"  Returning None - caller should use fallback (e.g., linear_decay)")
        return None

    if failed_batches > 0:
        print(f"  Fisher computation completed with {failed_batches} failed batches ({successful_batches} succeeded)")

    # Compute traces (sum of diagonal)
    fisher_traces = {}
    for name, fisher in fisher_accum.items():
        trace = (fisher / max(count, 1)).sum().item()
        fisher_traces[name] = trace

    print(f"Computed Fisher traces for {len(fisher_traces)} parameters from {count} samples")

    # Cache if path provided and we have valid results
    if cache_path and fisher_traces:
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


def generate_random_vector(
    task_vector: dict[str, torch.Tensor],
    seed: Optional[int] = None,
    match_norm: str = "per_param",
    cache_path: Optional[str] = None,
) -> dict[str, torch.Tensor]:
    """Generate a random vector with the same structure as the task vector.

    This baseline tests whether the overthinking effect is specific to the
    task vector direction (R - M) or if any random perturbation of similar
    magnitude produces similar effects.

    Args:
        task_vector: The original task vector to match structure/magnitude
        seed: Random seed for reproducibility. If None, uses random seed.
        match_norm: How to match the magnitude:
            - "per_param": Match L2 norm of each parameter individually
            - "global": Match total L2 norm across all parameters
            - "none": Use standard normal without scaling
        cache_path: Optional path to cache/load the random vector

    Returns:
        Dictionary mapping parameter name -> random tensor with matched norm
    """
    # Check cache
    if cache_path and Path(cache_path).exists():
        print(f"Loading cached random vector from: {cache_path}")
        try:
            random_vector = torch.load(cache_path, map_location="cpu", weights_only=True)
            if random_vector and len(random_vector) > 0:
                return random_vector
        except Exception as e:
            print(f"  Warning: Cache corrupted, regenerating: {e}")
            Path(cache_path).unlink(missing_ok=True)

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    random_vector = {}

    if match_norm == "per_param":
        # Match norm of each parameter individually
        for name, tv in task_vector.items():
            # Generate random tensor with same shape
            rand_tensor = torch.randn_like(tv)
            # Scale to match the original parameter's L2 norm
            tv_norm = torch.norm(tv).item()
            rand_norm = torch.norm(rand_tensor).item()
            if rand_norm > 0:
                rand_tensor = rand_tensor * (tv_norm / rand_norm)
            random_vector[name] = rand_tensor

    elif match_norm == "global":
        # First pass: generate random tensors
        for name, tv in task_vector.items():
            random_vector[name] = torch.randn_like(tv)

        # Compute global norms
        tv_global_norm = np.sqrt(sum(
            torch.norm(tv).item() ** 2 for tv in task_vector.values()
        ))
        rand_global_norm = np.sqrt(sum(
            torch.norm(rv).item() ** 2 for rv in random_vector.values()
        ))

        # Scale all random tensors uniformly
        if rand_global_norm > 0:
            scale = tv_global_norm / rand_global_norm
            random_vector = {
                name: rv * scale for name, rv in random_vector.items()
            }

    else:  # match_norm == "none"
        # Standard normal without scaling
        for name, tv in task_vector.items():
            random_vector[name] = torch.randn_like(tv)

    print(f"Generated random vector for {len(random_vector)} parameters (match_norm={match_norm})")

    # Report statistics
    tv_norms = [torch.norm(tv).item() for tv in task_vector.values()]
    rv_norms = [torch.norm(rv).item() for rv in random_vector.values()]
    print(f"  Task vector norm: total={sum(tv_norms):.2f}, mean={np.mean(tv_norms):.4f}")
    print(f"  Random vector norm: total={sum(rv_norms):.2f}, mean={np.mean(rv_norms):.4f}")

    # Cache if path provided
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(random_vector, cache_path)
        print(f"Cached random vector to: {cache_path}")

    return random_vector


def is_random_method(method: str) -> bool:
    """Check if a method name indicates random baseline.

    Args:
        method: Method name string

    Returns:
        True if this is a random baseline method
    """
    return method.startswith("random")


def parse_random_method(method: str) -> dict:
    """Parse random method string into parameters.

    Supports formats:
        - "random": default (seed=42, match_norm="per_param")
        - "random_seed123": specific seed
        - "random_global": global norm matching
        - "random_seed123_global": both

    Args:
        method: Method name string (e.g., "random_seed42_global")

    Returns:
        Dictionary with keys: seed, match_norm
    """
    params = {
        "seed": 42,  # Default seed for reproducibility
        "match_norm": "per_param",  # Default norm matching
    }

    if method == "random":
        return params

    parts = method.split("_")
    for part in parts[1:]:  # Skip "random" prefix
        if part.startswith("seed"):
            try:
                params["seed"] = int(part[4:])
            except ValueError:
                pass
        elif part in ["per_param", "global", "none"]:
            params["match_norm"] = part

    return params


# Registry of all coefficient methods
COEFFICIENT_METHODS = {
    "uniform": uniform_coefficients,
    "freeze_last_5": lambda n, **kw: freeze_last_n_coefficients(n, 5, **kw),
    "freeze_last_half": freeze_last_half_coefficients,
    "linear_decay": linear_decay_coefficients,
    "cosine_decay": cosine_decay_coefficients,
    "fisher": fisher_weighted_coefficients,
    "directional_fisher": directional_fisher_coefficients,
    "task_magnitude": task_magnitude_coefficients,
}
