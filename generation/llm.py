"""
generation/llm.py
------------------
Local LLM generation. Supports TinyLlama, Phi-2, Mistral.
Auto-detects model family and uses the correct prompt format.
"""

import os
import time
from typing import Optional
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv("LLM_MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
USE_4BIT = os.getenv("USE_4BIT", "false").lower() == "true"

# Per-model context window sizes
MODEL_CONTEXT_LIMITS = {
    "mistral":   4096,
    "tinyllama": 2048,
    "phi":       2048,
}

MAX_NEW_TOKENS = 256  # single source of truth — never set max_length alongside this


def _get_model_max_tokens(model_name: str) -> int:
    name = model_name.lower()
    for key, limit in MODEL_CONTEXT_LIMITS.items():
        if key in name:
            return limit
    return 2048  # safe default


class LLMGenerator:
    def __init__(self, model_name: str = MODEL_NAME, use_4bit: bool = USE_4BIT):
        self.model_name = model_name
        self.use_4bit = use_4bit
        self.model = None
        self.tokenizer = None
        self.model_max_tokens = _get_model_max_tokens(model_name)
        # Budget: reserve space for generated tokens + a small safety margin
        self.max_prompt_tokens = self.model_max_tokens - MAX_NEW_TOKENS - 64
        logger.info(
            f"Token budget | model_max={self.model_max_tokens} "
            f"prompt_budget={self.max_prompt_tokens} max_new={MAX_NEW_TOKENS}"
        )
        self._load_model()

    def _load_model(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        logger.info(f"Loading model: {self.model_name} | 4-bit={self.use_4bit}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map={"": 0},
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map={"": 0},
                trust_remote_code=True,
            )

        self.model.eval()
        device = next(self.model.parameters()).device
        logger.success(f"Model loaded on {device}.")

    # ─────────────────────────────────────────
    # Token-aware chunk truncation
    # ─────────────────────────────────────────

    def _truncate_chunks(self, chunks: list, token_budget: int) -> list:
        """
        Greedily include chunks until the token budget is exhausted.
        Ensures the context always fits within the model's window.
        """
        kept, total = [], 0
        for chunk in chunks:
            chunk_text = f"[Source: {chunk.paper_title}]\n{chunk.text}"
            tokens = len(self.tokenizer.encode(chunk_text, add_special_tokens=False))
            if total + tokens > token_budget:
                logger.debug(
                    f"Chunk truncated at {total}/{token_budget} tokens "
                    f"(dropped {len(chunks) - len(kept)} chunks)"
                )
                break
            kept.append(chunk)
            total += tokens
        return kept

    # ─────────────────────────────────────────
    # Generation
    # ─────────────────────────────────────────

    def generate(
        self,
        query: str,
        context_chunks: Optional[list] = None,
        temperature: float = 0.1,
    ) -> dict:
        """
        Generate an answer. max_new_tokens is fixed at module level to avoid
        the max_new_tokens + max_length conflict in HuggingFace generate().
        """
        import torch

        mode = "rag" if context_chunks else "baseline"

        # Estimate overhead: system prompt + question + assistant header
        overhead_tokens = len(self.tokenizer.encode(query, add_special_tokens=False)) + 128
        chunk_budget = self.max_prompt_tokens - overhead_tokens

        # Truncate chunks to fit within budget before building the prompt
        if context_chunks:
            context_chunks = self._truncate_chunks(context_chunks, chunk_budget)

        prompt = self._build_prompt(query, context_chunks)

        # Tokenize WITHOUT max_length — we already ensured it fits
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=False,   # no silent truncation; we handled it above
        ).to(self.model.device)

        prompt_tokens = inputs["input_ids"].shape[-1]

        # Warn if somehow over budget (shouldn't happen, but good to know)
        if prompt_tokens > self.max_prompt_tokens:
            logger.warning(
                f"Prompt ({prompt_tokens}) exceeds budget ({self.max_prompt_tokens}). "
                "Falling back to tokenizer truncation."
            )
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_prompt_tokens,
            ).to(self.model.device)
            prompt_tokens = inputs["input_ids"].shape[-1]

        logger.debug(f"Prompt tokens: {prompt_tokens} | Max new: {MAX_NEW_TOKENS}")

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,   # ← only this, never max_length
                temperature=max(temperature, 1e-4),
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )
        generation_ms = (time.perf_counter() - t0) * 1000

        new_tokens = outputs[0][prompt_tokens:]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        completion_tokens = len(new_tokens)

        return {
            "answer": answer,
            "mode": mode,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "generation_ms": round(generation_ms, 2),
        }

    # ─────────────────────────────────────────
    # Prompt builder
    # ─────────────────────────────────────────

    def _build_prompt(self, query: str, context_chunks: Optional[list]) -> str:
        """Build prompt — works for TinyLlama, Phi-2, Mistral."""

        if context_chunks:
            context_text = "\n\n---\n\n".join([
                f"[Source {i+1}: {c.paper_title}]\n{c.text}"
                for i, c in enumerate(context_chunks)
            ])
            system = (
                "You are a precise scientific assistant. "
                "Answer the question using ONLY the provided context. "
                "If the context does not contain enough information, say so. "
                "Do not hallucinate or use external knowledge."
            )
            user_content = (
                f"Context:\n{context_text}\n\n"
                f"Question: {query}\n\n"
                f"Answer (based only on the context above):"
            )
        else:
            system = (
                "You are a knowledgeable scientific assistant. "
                "Answer the question accurately and concisely."
            )
            user_content = f"Question: {query}\n\nAnswer:"

        prompt = f"### System:\n{system}\n\n### User:\n{user_content}\n\n### Assistant:"
        return prompt