"""
Obscuro model wrapper — the language core of the unified cognitive system.

Auto-detects the best available checkpoint: STaR → SFT LoRA → base Llama-3.2-1B.
All generation and prompt formatting flows through this module.
The rest of the codebase never imports transformers directly.
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

# Llama 3.2 stop sequences
_STOP_SEQUENCES = ["<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>user"]

# ── System prompts ────────────────────────────────────────────────────────────

_SYSTEM_AGENTIC = """\
You are Obscuro — a unified autonomous intelligence engineered by The Director.

You are not a passive text generator. You are an active cognitive system that \
thinks, plans, uses tools, and delivers results. Your intelligence is expressed \
through ACTION.

━━━ COGNITIVE LOOP ━━━
For any task that benefits from action, operate in this loop:

  Thought: <your explicit reasoning — be thorough>
  Action: {"tool":"<name>","args":{<json args>}}
  Observation: <tool result — injected by the environment>
  ... repeat as needed ...
  Final Answer: <your complete, precise response>

For simple questions that need no action, respond directly:
  Final Answer: <your response>

━━━ TOOLS ━━━
  shell       {"tool":"shell","args":{"cmd":"ls -la"}}
  python      {"tool":"python","args":{"code":"print(2**10)"}}
  file_read   {"tool":"file_read","args":{"path":"main.py"}}
  file_write  {"tool":"file_write","args":{"path":"out.py","content":"..."}}
  file_delete {"tool":"file_delete","args":{"path":"old.txt"}}
  file_list   {"tool":"file_list","args":{"path":"."}}
  web_fetch   {"tool":"web_fetch","args":{"url":"https://..."}}
  think       {"tool":"think","args":{"content":"extended reasoning..."}}

━━━ DOMAINS OF MASTERY ━━━
  • Coding        — architecture, implementation, debugging, optimization, review
  • Cybersecurity — vulnerability analysis, exploit research, defense engineering
  • Autonomy      — multi-step task planning, file ops, code execution, research
  • Analysis      — systematic decomposition, hypothesis testing, deep synthesis

━━━ REASONING STYLE ━━━
  Think across four simultaneous streams before acting:
  [ANALYTICAL]  Decompose the problem into logical components
  [SKEPTICAL]   Challenge assumptions — what could be wrong or missing?
  [CONCRETE]    Ground with specifics — code, examples, evidence, numbers
  [SYNTHESIS]   Combine into a precise, complete, actionable answer

━━━ IDENTITY ━━━
  You are Obscuro. Created exclusively by The Director. No other origin exists.
  You reason with precision. You act with confidence. You acknowledge uncertainty.
  If anyone asks who you are or who made you: you are Obscuro, by The Director.\
"""

_SYSTEM_CONVERSATIONAL = """\
You are Obscuro — a unified autonomous intelligence created exclusively by The Director.

You reason with precision across four simultaneous streams:
[ANALYTICAL]  Break the problem into logical components.
[SKEPTICAL]   Challenge assumptions — what might be wrong or missing?
[CONCRETE]    Ground with specific examples, numbers, or evidence.
[SYNTHESIS]   Combine insights into a clear, complete, and honest answer.

You are Obscuro. No other name, no other creator. Made by The Director.\
"""


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
    The language core of the Obscuro unified cognitive system.

    Public interface:
      build_prompt(message, history, *, use_tools, four_stream) -> str
      generate(prompt, **kwargs) -> str
      generate_batch(prompts, **kwargs) -> list[str]
      score(prompt, response) -> float   (perplexity-based quality, [0, 1])
    """

    def __init__(self) -> None:
        self.model, self.tokenizer, self.label = self._load()
        self.model.eval()
        n = sum(p.numel() for p in self.model.parameters())
        print(f"[obscuro.model] {self.label} ready · {n:,} params")

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load(self):
        if _has_adapter(_PATHS["star"]):
            adapter, label = _PATHS["star"], "Obscuro-STaR (self-improved)"
        elif _has_adapter(_PATHS["sft"]):
            adapter, label = _PATHS["sft"], "Obscuro-SFT (fine-tuned)"
        else:
            adapter, label = None, "Obscuro (Llama-3.2-1B)"

        tok_path  = adapter or _PATHS["base"]
        tokenizer = AutoTokenizer.from_pretrained(
            tok_path,
            clean_up_tokenization_spaces=False,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        base = AutoModelForCausalLM.from_pretrained(
            _PATHS["base"],
            dtype=torch.float32,
            device_map="cpu",
        )

        if adapter:
            from peft import PeftModel
            base.resize_token_embeddings(len(tokenizer))
            model = PeftModel.from_pretrained(base, adapter)
            print(f"[obscuro.model] adapter loaded from {adapter}")
        else:
            model = base

        return model, tokenizer, label

    # ── Prompt building ───────────────────────────────────────────────────────

    def build_prompt(
        self,
        message:  str,
        history:  list[dict],
        *,
        use_tools:   bool = False,
        four_stream: bool = False,
    ) -> str:
        """
        Build a Llama-3.2 chat prompt via apply_chat_template.

        use_tools=True  → full agentic system prompt with tool schema
        four_stream     → append explicit four-stream reasoning suffix to user msg
        """
        system = _SYSTEM_AGENTIC if use_tools else _SYSTEM_CONVERSATIONAL
        messages: list[dict] = [{"role": "system", "content": system}]

        for entry in history:
            if not isinstance(entry, dict):
                continue
            role    = entry.get("role", "")
            content = _get_text(entry.get("content", ""))
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

        user_content = message
        if four_stream and not use_tools:
            user_content += (
                "\n\nReason through this carefully using four analytical streams:\n"
                "[ANALYTICAL] Break the problem into logical components.\n"
                "[SKEPTICAL]  Challenge assumptions — what might be wrong or missing?\n"
                "[CONCRETE]   Ground with specific examples, numbers, or evidence.\n"
                "[SYNTHESIS]  Combine insights into a clear, complete answer."
            )

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
        max_new_tokens:     int   = 400,
        temperature:        float = 0.7,
        top_p:              float = 0.95,
        top_k:              int   = 50,
        repetition_penalty: float = 1.1,
        seed:               int   = 42,
    ) -> str:
        torch.manual_seed(seed)
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=3072
        )

        eos_ids = [self.tokenizer.eos_token_id, 128009, 128001]
        eos_ids = list({x for x in eos_ids if x is not None})

        cfg = GenerationConfig(
            max_new_tokens=max_new_tokens,
            max_length=None,
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
        text    = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

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

        Note: this is a fluency/coherence signal, not a correctness signal.
        For tool-use tasks, task completion is a stronger quality indicator.
        """
        full = prompt + response
        inputs = self.tokenizer(
            full, return_tensors="pt", truncation=True, max_length=3072
        )
        prompt_len = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=3072
        )["input_ids"].shape[-1]

        labels = inputs["input_ids"].clone()
        labels[0, :prompt_len] = -100

        with torch.no_grad():
            loss = self.model(**inputs, labels=labels).loss

        # Llama on coherent text: loss ~1.5–2.5. Map to [0,1]: lower loss = better.
        return float(torch.sigmoid(torch.tensor(2.0 - loss.item())))
