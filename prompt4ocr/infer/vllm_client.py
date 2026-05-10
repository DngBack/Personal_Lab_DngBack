from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VllmGenerationConfig:
    """Configuration for a single vLLM text generation call."""

    temperature: float = 0.1
    top_p: float = 0.95
    max_tokens: int = 2048


class VllmTextGenerator:
    """Simple wrapper around vLLM offline generation APIs."""

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        trust_remote_code: bool = True,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.90,
    ) -> None:
        """Initialize vLLM model client.

        Args:
            model_name: HuggingFace model id or local model path.
            tensor_parallel_size: Number of GPUs for tensor parallelism.
            trust_remote_code: Whether to trust model repo custom code.
            max_model_len: Max sequence length reserved for KV cache.
            gpu_memory_utilization: Fraction of GPU memory reserved by vLLM.
        """
        try:
            from vllm import LLM
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Missing dependency 'vllm'. Install it before running inference."
            ) from exc
        self._llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=trust_remote_code,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )

    def generate_text(self, prompt: str, config: VllmGenerationConfig) -> str:
        """Generate one response text from prompt with sampling config."""
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
        )
        outputs = self._llm.generate(prompt, sampling_params=sampling_params)
        if not outputs or not outputs[0].outputs:
            raise RuntimeError("No generation returned from vLLM.")
        return outputs[0].outputs[0].text
