"""Model loading and task vector utilities."""

import torch
import gc
import re
from pathlib import Path
from typing import Optional


def monitor_gpu_memory(label: str = "") -> tuple[float, float]:
    """Monitor and log current GPU memory usage.

    Args:
        label: Optional label for the log message

    Returns:
        Tuple of (allocated_gb, reserved_gb)
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"[MEM {label}] Allocated: {allocated:.1f}GB, Reserved: {reserved:.1f}GB")
        return allocated, reserved
    print(f"[MEM {label}] CUDA not available")
    return 0.0, 0.0


def force_cleanup():
    """Aggressive memory cleanup - garbage collect and clear CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_model_for_inference(
    model_path: str,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
    trust_remote_code: bool = True,
) -> tuple:
    """Load model and tokenizer for inference.

    Args:
        model_path: Path to model (local or HuggingFace)
        device: Device to load model on
        dtype: Data type for model weights
        trust_remote_code: Whether to trust remote code

    Returns:
        Tuple of (model, tokenizer)
    """
    from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

    print(f"Loading model from: {model_path}")
    monitor_gpu_memory("before load")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model.eval()

    print(f"Model loaded: {model.num_parameters():,} parameters")
    monitor_gpu_memory("after load")

    return model, tokenizer


def compute_task_vector(
    reasoning_model_path: str,
    instruct_model_path: str,
    dtype: torch.dtype = torch.bfloat16,
    cache_path: Optional[str] = None,
) -> dict[str, torch.Tensor]:
    """Compute task vector (R - M) on CPU to save GPU memory.

    Args:
        reasoning_model_path: Path to reasoning (thinking) model
        instruct_model_path: Path to instruct model
        dtype: Data type for computation
        cache_path: Optional path to cache/load task vector

    Returns:
        Dictionary mapping parameter names to task vector tensors
    """
    from transformers import Qwen3VLForConditionalGeneration

    # Check for cached task vector
    if cache_path and Path(cache_path).exists():
        print(f"Loading cached task vector from: {cache_path}")
        task_vector = torch.load(cache_path, map_location="cpu")
        print(f"Loaded task vector: {len(task_vector)} parameters")
        return task_vector

    print(f"Loading R (reasoning) model to CPU: {reasoning_model_path}")
    model_R = Qwen3VLForConditionalGeneration.from_pretrained(
        reasoning_model_path,
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=True
    )

    print(f"Loading M (instruct) model to CPU: {instruct_model_path}")
    model_M = Qwen3VLForConditionalGeneration.from_pretrained(
        instruct_model_path,
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=True
    )

    print("Computing task vector (R - M)...")
    task_vector = {}
    R_params = dict(model_R.named_parameters())
    M_params = dict(model_M.named_parameters())

    matched = 0
    for name in R_params:
        if name in M_params:
            if R_params[name].shape == M_params[name].shape:
                task_vector[name] = (R_params[name] - M_params[name]).cpu()
                matched += 1

    del model_R, model_M, R_params, M_params
    force_cleanup()

    print(f"Task vector computed: {matched} parameters")

    # Cache if path provided
    if cache_path:
        print(f"Caching task vector to: {cache_path}")
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(task_vector, cache_path)

    return task_vector


def get_layer_index(param_name: str) -> Optional[int]:
    """Extract layer index from parameter name.

    Args:
        param_name: Full parameter name (e.g., "model.layers.15.self_attn.q_proj.weight")

    Returns:
        Layer index as int, or None if not a layer parameter
    """
    match = re.search(r'layers\.(\d+)\.', param_name)
    return int(match.group(1)) if match else None


def apply_task_vector(
    model,
    task_vector: dict[str, torch.Tensor],
    alpha: float,
    coefficients: Optional[dict[str, float]] = None,
):
    """Apply task vector to model with optional layer-wise coefficients.

    Formula: param = param + alpha * coef * task_vector[param]

    Args:
        model: The model to modify
        task_vector: Dictionary of parameter name -> delta tensor
        alpha: Scaling factor for task vector
        coefficients: Optional per-parameter coefficients (default 1.0)
    """
    if alpha == 0.0:
        return

    applied = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in task_vector:
                coef = coefficients.get(name, 1.0) if coefficients else 1.0
                if coef == 0.0:
                    continue
                delta = task_vector[name].to(param.device, dtype=param.dtype)
                param.add_(alpha * coef * delta)
                applied += 1

    print(f"Applied task vector (alpha={alpha}) to {applied} parameters")


def remove_task_vector(
    model,
    task_vector: dict[str, torch.Tensor],
    alpha: float,
    coefficients: Optional[dict[str, float]] = None,
):
    """Remove previously applied task vector from model.

    Formula: param = param - alpha * coef * task_vector[param]

    Args:
        model: The model to modify
        task_vector: Dictionary of parameter name -> delta tensor
        alpha: Scaling factor that was used when applying
        coefficients: Optional per-parameter coefficients (default 1.0)
    """
    if alpha == 0.0:
        return

    removed = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in task_vector:
                coef = coefficients.get(name, 1.0) if coefficients else 1.0
                if coef == 0.0:
                    continue
                delta = task_vector[name].to(param.device, dtype=param.dtype)
                param.sub_(alpha * coef * delta)
                removed += 1

    print(f"Removed task vector (alpha={alpha}) from {removed} parameters")