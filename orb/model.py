"""
Orb model wrapper — loading, generation, and perplexity-based scoring.

Auto-detects the best available checkpoint: STaR → SFT LoRA → base GPT-2.
All generation goes through this module so the rest of the codebase never
touches transformers directly.
"""
from __future__ import annotations

import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GenerationConfig

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_PATHS = {
    "star": os.path.join(_ROOT, "models", "orb-star"),
    "sft":  os.path.join(_ROOT, "models", "orb-sft-lora"),
    "base": os.path.join(_ROOT, "models", "gpt2"),
}

_SPECIAL_TOKENS = ["<|system|>", "<|user|>", "<|orb|>", "<|end|>"]

_STOP_SEQUENCES = ["<|user|>", "<|end|>", "User:", "Human:", "\nUser", "\nHuman"]


def _has_adapter(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "adapter_config.json"))


class OrbModel:
    """
    Thin wrapper around GPT-2 (+ optional LoRA adapter).

    Public interface:
      generate(prompt, **kwargs) -> str
      generate_batch(prompts, **kwargs) -> list[str]   (sequential; GPT-2 is CPU)
      score(prompt, response) -> float                  (perplexity-based, [0,1])
    """

    def __init__(self) -> None:
        self.model, self.tokenizer, self.label = self._load()
        self.model.eval()
        n = sum(p.numel() for p in self.model.parameters())
        print(f"[orb.model] {self.label} ready · {n:,} params")

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load(self):
        if _has_adapter(_PATHS["star"]):
            adapter, label = _PATHS["star"], "Orb-STaR (self-improved)"
        elif _has_adapter(_PATHS["sft"]):
            adapter, label = _PATHS["sft"], "Orb-SFT (fine-tuned)"
        else:
            adapter, label = None, "Orb Base (GPT-2)"

        tok_path = adapter or _PATHS["base"]
        tokenizer = GPT2Tokenizer.from_pretrained(tok_path)
        tokenizer.pad_token = tokenizer.eos_token

        base = GPT2LMHeadModel.from_pretrained(_PATHS["base"])

        if adapter:
            from peft import PeftModel
            tokenizer.add_special_tokens({"additional_special_tokens": _SPECIAL_TOKENS})
            base.resize_token_embeddings(len(tokenizer))
            model = PeftModel.from_pretrained(base, adapter)
            print(f"[orb.model] adapter loaded from {adapter}")
        else:
            model = base

        return model, tokenizer, label

    # ── Generation ────────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 200,
        temperature: float = 0.85,
        top_p: float = 0.95,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        seed: int = 42,
    ) -> str:
        torch.manual_seed(seed)
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=880
        )
        cfg = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        with torch.no_grad():
            out = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                generation_config=cfg,
            )
        new_ids = out[0][inputs["input_ids"].shape[-1]:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        for stop in _STOP_SEQUENCES:
            if stop in text:
                text = text[: text.index(stop)].strip()
        return text

    def generate_batch(self, prompts: list[str], **kwargs) -> list[str]:
        """Run generate() for each prompt. Sequential on CPU — no batching overhead."""
        return [self.generate(p, **kwargs) for p in prompts]

    # ── Scoring ───────────────────────────────────────────────────────────────

    def score(self, prompt: str, response: str) -> float:
        """
        Estimate response quality via per-token cross-entropy on response tokens only.
        Returns a value in [0, 1] where higher is better.

        Uses the model's own likelihood as a proxy for coherence — a response
        that flows naturally from the prompt scores higher than one that doesn't.
        """
        full = prompt + response
        inputs = self.tokenizer(
            full, return_tensors="pt", truncation=True, max_length=1024
        )
        prompt_ids = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        )["input_ids"]
        prompt_len = prompt_ids.shape[-1]

        labels = inputs["input_ids"].clone()
        labels[0, :prompt_len] = -100  # mask prompt tokens from loss

        with torch.no_grad():
            loss = self.model(**inputs, labels=labels).loss

        # Map cross-entropy loss to [0, 1]: lower loss → higher score
        # loss of ~3 is typical for GPT-2 on coherent text
        return float(torch.sigmoid(torch.tensor(3.0 - loss)).item())
