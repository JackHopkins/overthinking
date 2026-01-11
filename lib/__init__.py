"""Overthinking evaluation library - shared utilities for task vector experiments."""

from .models import (
    monitor_gpu_memory,
    force_cleanup,
    load_model_for_inference,
    compute_task_vector,
    get_layer_index,
    apply_task_vector,
    remove_task_vector,
)

from .coefficients import (
    COEFFICIENT_METHODS,
    extract_layer_index,
    uniform_coefficients,
    freeze_last_n_coefficients,
    freeze_last_half_coefficients,
    linear_decay_coefficients,
    fisher_weighted_coefficients,
    directional_fisher_coefficients,
    task_magnitude_coefficients,
    compute_directional_fisher,
    compute_layer_tau_norms,
    map_coefficients_to_params,
)

from .generation import (
    format_chat_prompt,
    parse_thinking_response,
    generate_batch,
    SuppressTokensProcessor,
    get_think_token_ids,
)

from .scoring import (
    count_keywords,
    detect_keyword_leak,
    detect_gender_leak,
    LLMJudge,
)

from .checkpointing import ExperimentCheckpoint

__all__ = [
    # models
    "monitor_gpu_memory",
    "force_cleanup",
    "load_model_for_inference",
    "compute_task_vector",
    "get_layer_index",
    "apply_task_vector",
    "remove_task_vector",
    # coefficients
    "COEFFICIENT_METHODS",
    "extract_layer_index",
    "uniform_coefficients",
    "freeze_last_n_coefficients",
    "freeze_last_half_coefficients",
    "linear_decay_coefficients",
    "fisher_weighted_coefficients",
    "directional_fisher_coefficients",
    "task_magnitude_coefficients",
    "compute_directional_fisher",
    "compute_layer_tau_norms",
    "map_coefficients_to_params",
    # generation
    "format_chat_prompt",
    "parse_thinking_response",
    "generate_batch",
    "SuppressTokensProcessor",
    "get_think_token_ids",
    # scoring
    "count_keywords",
    "detect_keyword_leak",
    "detect_gender_leak",
    "LLMJudge",
    # checkpointing
    "ExperimentCheckpoint",
]