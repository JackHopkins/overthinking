"""Batch generation with <think> parsing and token suppression."""

import torch
from typing import Optional
from transformers import LogitsProcessor, LogitsProcessorList


class SuppressTokensProcessor(LogitsProcessor):
    """Apply a large negative bias to specific token IDs to prevent sampling.

    Used to suppress repeated <think> tokens after prefilling.
    """

    def __init__(self, suppress_token_ids: list[int], bias: float = -100.0):
        """Initialize the processor.

        Args:
            suppress_token_ids: List of token IDs to suppress
            bias: Negative bias to apply (default -100.0 effectively prevents sampling)
        """
        self.suppress_token_ids = suppress_token_ids
        self.bias = bias

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Apply negative bias to suppressed tokens.

        Args:
            input_ids: Current input token IDs
            scores: Logits for next token prediction

        Returns:
            Modified scores with suppressed tokens biased
        """
        for token_id in self.suppress_token_ids:
            if token_id < scores.shape[-1]:
                scores[:, token_id] = self.bias
        return scores


def get_think_token_ids(tokenizer) -> list[int]:
    """Get token IDs for <think> and related tokens to suppress.

    Args:
        tokenizer: The tokenizer to use for encoding

    Returns:
        List of unique token IDs that represent think-related tokens
    """
    tokens_to_suppress = ["<think>", "<think", "think>", " <think>", "<think> "]
    token_ids = []

    for token in tokens_to_suppress:
        try:
            ids = tokenizer.encode(token, add_special_tokens=False)
            token_ids.extend(ids)
        except Exception:
            pass

    return list(set(token_ids))


def format_chat_prompt(
    user_message: str,
    prefill_think: bool = True,
    system_message: Optional[str] = None,
) -> str:
    """Format user message as chat prompt with optional <think> prefill.

    Args:
        user_message: The user's message
        prefill_think: Whether to add <think> to trigger reasoning mode
        system_message: Optional system message to prepend

    Returns:
        Formatted chat prompt string
    """
    prompt = ""

    if system_message:
        prompt += f"<|im_start|>system\n{system_message}<|im_end|>\n"

    prompt += f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"

    if prefill_think:
        prompt += "<think>"

    return prompt


def parse_thinking_response(text: str) -> dict[str, str]:
    """Parse response to extract thinking and final response.

    Splits on </think> tag to separate internal reasoning from output.
    Also handles cases where the model echoes input - anything before
    'assistant\n' is discarded as it marks where sampling begins.

    Args:
        text: Full generated text

    Returns:
        Dictionary with 'thinking', 'response', and 'full_output' keys
    """
    original_text = text

    # Discard anything before "assistant\n" - this is echoed input, not generation
    if "assistant\n" in text:
        text = text.split("assistant\n", 1)[1]

    # Remove leading <think> tag if present (from prefill echo)
    text = text.lstrip()
    if text.startswith("<think>"):
        text = text[7:]

    thinking = ""
    response = text

    if "</think>" in text:
        parts = text.split("</think>", 1)
        thinking = parts[0].strip()
        response = parts[1].strip() if len(parts) > 1 else ""

    return {
        "thinking": thinking,
        "response": response,
        "full_output": original_text,
    }


def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    do_sample: bool = True,
    suppress_think: bool = True,
    suppress_think_bias: float = -100.0,
    batch_size: Optional[int] = None,
) -> list[dict]:
    """Generate responses for a batch of prompts.

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompts: List of formatted prompts
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        do_sample: Whether to sample (vs greedy)
        suppress_think: Whether to suppress <think> tokens after prefill
        suppress_think_bias: Bias to apply for suppression
        batch_size: If provided, process in sub-batches to manage memory

    Returns:
        List of dicts with 'thinking', 'response', 'full_output', 'prompt' keys
    """
    if batch_size and len(prompts) > batch_size:
        # Process in sub-batches
        all_results = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            batch_results = generate_batch(
                model, tokenizer, batch, max_new_tokens, temperature,
                do_sample, suppress_think, suppress_think_bias, None
            )
            all_results.extend(batch_results)
        return all_results

    # Tokenize batch with padding
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(model.device)

    # Create logits processor to suppress <think> tokens
    logits_processor = None
    if suppress_think:
        think_ids = get_think_token_ids(tokenizer)
        if think_ids:
            logits_processor = LogitsProcessorList([
                SuppressTokensProcessor(think_ids, bias=suppress_think_bias)
            ])

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
            logits_processor=logits_processor,
        )

    # Decode and parse each response
    results = []
    for i, output in enumerate(outputs):
        # Calculate input length (accounting for padding)
        input_len = (inputs.attention_mask[i] == 1).sum().item()

        # Decode generated tokens only
        generated = tokenizer.decode(output[input_len:], skip_special_tokens=True)

        # Parse thinking vs response
        parsed = parse_thinking_response(generated)
        parsed["prompt"] = prompts[i]

        results.append(parsed)

    return results


def generate_single(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    do_sample: bool = True,
    suppress_think: bool = True,
) -> dict:
    """Generate a single response (convenience wrapper).

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Single formatted prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        do_sample: Whether to sample
        suppress_think: Whether to suppress <think> tokens

    Returns:
        Dict with 'thinking', 'response', 'full_output', 'prompt' keys
    """
    results = generate_batch(
        model, tokenizer, [prompt],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        suppress_think=suppress_think,
    )
    return results[0]