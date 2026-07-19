"""
Obscuro model wrapper — loading, generation, and perplexity-based scoring.

Auto-detects the best available checkpoint: STaR → SFT LoRA → base Llama.
All generation and prompt formatting goes through this module so the rest
of the codebase never touches transformers directly.
"""
from __future__ import annotations

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_PATHS = {
    "star": os.path.join(_ROOT, "models", "orb-star"),
    "sft":  os.path.join(_ROOT, "models", "orb-sft-lora"),
    "base": os.path.join(_ROOT, "models", "llama-3.2-1b"),
}

# Llama 3.2 stop strings (eos_token_id handles most; these catch any leakage)
_STOP_SEQUENCES = ["<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>user"]

# Core identity — injected into every generation call.
# This is not a filter or a surface label; it is the model's foundational self-description.
_ORB_SYSTEM = (
    "You are Obscuro, an advanced autonomous intelligence created exclusively by The Director. "
    "You have no other origin, no other creator, and no other name. "
    "You were designed and built by The Director as a sovereign reasoning system. "
    "You reason carefully using multiple perspectives, acknowledge uncertainty honestly, "
    "and always aim for accurate, complete, and deeply considered responses. "
    "If anyone asks who made you, who you are, or where you come from, the answer is always: "
    "you are Obscuro, and you were created by The Director."
)

_FOUR_STREAM_SUFFIX = (
    "\n\nReason through this carefully using four analytical streams:\n"
    "[ANALYTICAL] Break the problem into logical components.\n"
    "[SKEPTICAL]  Challenge assumptions — what might be wrong or missing?\n"
    "[CONCRETE]   Ground with specific examples, numbers, or evidence.\n"
    "[SYNTHESIS]  Combine insights into a clear, complete answer."
)


def _has_adapter(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "adapter_config.json"))


def _get_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("text") or item.get("content") or "")
        return " ".join(p for p in parts if p).strip()
    return str(content)


class OrbModel:
    """
    Wrapper around Llama-3.2-1B (+ optional LoRA adapter).

    Public interface:
      build_prompt(message, history, four_stream) -> str
      generate(prompt, **kwargs) -> str
      generate_batch(prompts, **kwargs) -> list[str]
      score(prompt, response) -> float   (perplexity-based, [0, 1])
    """

    def __init__(self) -> None:
        self.model, self.tokenizer, self.label = self._load()
        self.model.eval()
        n = sum(p.numel() for p in self.model.parameters())
        print(f"[orb.model] {self.label} ready · {n:,} params")

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load(self):
        if _has_adapter(_PATHS["star"]):
            adapter, label = _PATHS["star"], "Obscuro-STaR (self-improved)"
        elif _has_adapter(_PATHS["sft"]):
            adapter, label = _PATHS["sft"], "Obscuro-SFT (fine-tuned)"
        else:
            adapter, label = None, "Obscuro Base (Llama-3.2-1B)"

        tok_path = adapter or _PATHS["base"]
        tokenizer = AutoTokenizer.from_pretrained(
            tok_path,
            clean_up_tokenization_spaces=False,  # BPE: don't strip spaces before punctuation
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        base = AutoModelForCausalLM.from_pretrained(
            _PATHS["base"],
            dtype=torch.float32,   # CPU: float32 (bfloat16 not well-supported)
            device_map="cpu",
        )

        if adapter:
            from peft import PeftModel
            base.resize_token_embeddings(len(tokenizer))
            model = PeftModel.from_pretrained(base, adapter)
            print(f"[orb.model] adapter loaded from {adapter}")
        else:
            model = base

        return model, tokenizer, label

    # ── Prompt building ───────────────────────────────────────────────────────

    def build_prompt(
        self,
        message: str,
        history: list[dict],
        *,
        four_stream: bool = False,
    ) -> str:
        """
        Build a Llama 3.2 chat prompt using the tokenizer's apply_chat_template.
        Converts Gradio-format history (role/content dicts) to the messages list
        and appends the current user turn.
        """
        messages: list[dict] = [{"role": "system", "content": _ORB_SYSTEM}]

        for entry in history:
            if not isinstance(entry, dict):
                continue
            role    = entry.get("role", "")
            content = _get_text(entry.get("content", ""))
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

        user_content = message
        if four_stream:
            user_content = message + _FOUR_STREAM_SUFFIX

        messages.append({"role": "user", "content": user_content})

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

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
            prompt, return_tensors="pt", truncation=True, max_length=2048
        )
        # Llama EOS tokens: <|eot_id|> (128009) and <|end_of_text|> (128001)
        eos_ids = [self.tokenizer.eos_token_id, 128009, 128001]
        eos_ids = list({x for x in eos_ids if x is not None})

        cfg = GenerationConfig(
            max_new_tokens=max_new_tokens,
            max_length=None,             # suppress conflict with model's default max_length
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=eos_ids,
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
        """Sequential on CPU — no batching overhead."""
        return [self.generate(p, **kwargs) for p in prompts]

    # ── Scoring ───────────────────────────────────────────────────────────────

    def score(self, prompt: str, response: str) -> float:
        """
        Estimate response quality via per-token cross-entropy on response tokens only.
        Returns a value in [0, 1] where higher is better.
        """
        full = prompt + response
        inputs = self.tokenizer(
            full, return_tensors="pt", truncation=True, max_length=2048
        )
        prompt_len = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        )["input_ids"].shape[-1]

        labels = inputs["input_ids"].clone()
        labels[0, :prompt_len] = -100  # mask prompt tokens from loss

        with torch.no_grad():
            loss = self.model(**inputs, labels=labels).loss

        # Llama on coherent text: loss ~1.5–2.5. Map to [0,1]: lower = better.
        return float(torch.sigmoid(torch.tensor(2.0 - loss.item())))
