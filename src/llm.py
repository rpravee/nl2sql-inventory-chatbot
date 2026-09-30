"""LLM backends. Both expose the same `chat(messages) -> str` method."""
from __future__ import annotations

from typing import Protocol


class LLMConfigError(RuntimeError):
    """Raised when the LLM backend is not configured correctly (missing token, etc.)."""


class LLM(Protocol):
    def chat(self, messages: list[dict], max_tokens: int = 512, temperature: float = 0.1) -> str: ...


class HFInferenceLLM:
    """Calls a Llama model through Hugging Face Inference Providers (no GPU needed)."""

    def __init__(self, model: str, token: str | None):
        if not token:
            raise LLMConfigError(
                "HF_TOKEN is missing. Create a token at https://huggingface.co/settings/tokens "
                "and put it in your .env file."
            )
        from huggingface_hub import InferenceClient

        self.model = model
        self.client = InferenceClient(model=model, token=token)

    def chat(self, messages: list[dict], max_tokens: int = 512, temperature: float = 0.1) -> str:
        try:
            response = self.client.chat_completion(
                messages=messages, max_tokens=max_tokens, temperature=temperature
            )
        except Exception as exc:  # network, auth, gated model, quota ...
            raise LLMConfigError(
                f"Hugging Face request failed for {self.model}: {exc}\n"
                "Check that your token is valid, that you accepted the model license on its "
                "Hugging Face page, and that the model is available through Inference Providers."
            ) from exc
        return response.choices[0].message.content or ""


class LocalTransformersLLM:
    """Runs the model on your own machine with transformers (GPU recommended)."""

    def __init__(self, model: str, token: str | None):
        try:
            import torch
            from transformers import pipeline
        except ImportError as exc:
            raise LLMConfigError("Local backend needs: pip install -r requirements-local.txt") from exc

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.pipe = pipeline("text-generation", model=model, token=token,
                             torch_dtype=dtype, device_map="auto")

    def chat(self, messages: list[dict], max_tokens: int = 512, temperature: float = 0.1) -> str:
        out = self.pipe(messages, max_new_tokens=max_tokens, do_sample=False, return_full_text=False)
        generated = out[0]["generated_text"]
        if isinstance(generated, list):  # some versions return the message list
            generated = generated[-1]["content"]
        return generated


def get_llm(backend: str, model: str, token: str | None) -> LLM:
    if backend == "api":
        return HFInferenceLLM(model, token)
    if backend == "local":
        return LocalTransformersLLM(model, token)
    raise LLMConfigError(f"Unknown LLM_BACKEND '{backend}'. Use 'api' or 'local'.")
