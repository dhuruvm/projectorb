"""
Orb Model AI Chat — Gradio 6 · HuggingFace Transformers
OASIS Training Framework · Constitutional Self-Critique · Four-Stream Reasoning
"""

import os
import re
import traceback

import gradio as gr
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GenerationConfig

# ── Model loading — auto-detect best available checkpoint ────────────────────

BASE_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "gpt2")
STAR_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "orb-star")
SFT_PATH    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "orb-sft-lora")

SPECIAL_TOKENS = ["<|system|>", "<|user|>", "<|orb|>", "<|end|>"]

def _has_adapter(path):
    return os.path.isfile(os.path.join(path, "adapter_config.json"))

def load_model():
    if _has_adapter(STAR_PATH):
        adapter_path = STAR_PATH
        model_label  = "Orb-STaR (self-improved)"
    elif _has_adapter(SFT_PATH):
        adapter_path = SFT_PATH
        model_label  = "Orb-SFT (OASIS fine-tuned)"
    else:
        adapter_path = None
        model_label  = "Orb Base (pre-training)"

    tok_path = adapter_path or BASE_PATH
    print(f"[startup] Loading tokenizer from {tok_path} …")
    tokenizer = GPT2Tokenizer.from_pretrained(tok_path)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"[startup] Loading base weights …")
    base = GPT2LMHeadModel.from_pretrained(BASE_PATH)

    if adapter_path:
        tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
        base.resize_token_embeddings(len(tokenizer))
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, adapter_path)
        print(f"[startup] Loaded adapter: {adapter_path}")
    else:
        model = base

    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[startup] {model_label} ready · {n_params:,} params")
    return model, tokenizer, model_label

model, tokenizer, MODEL_LABEL = load_model()

# ── Orb constitution ──────────────────────────────────────────────────────────

ORB_SYSTEM = (
    "You are Orb, an advanced language model built on the OASIS framework. "
    "You reason carefully using multiple perspectives, acknowledge uncertainty honestly, "
    "and always aim for accurate, complete, and helpful responses."
)

FOUR_STREAM_SUFFIX = """

Reason through this carefully using four analytical streams before answering:
[ANALYTICAL] Break the problem into logical components.
[SKEPTICAL]  Challenge assumptions — what might be wrong or missing?
[CONCRETE]   Ground with specific examples, numbers, or evidence.
[SYNTHESIS]  Combine insights into a clear, complete answer.

"""

CRITIQUE_PROMPT_TEMPLATE = """\
Review this response and identify specific weaknesses:

Question: {question}
Response: {response}

Apply these checks — be critical and specific:
1. Accuracy — any claims that might be wrong or unverified?
2. Completeness — important aspects missing?
3. Clarity — would a non-expert understand it?
4. Honesty — is uncertainty appropriately expressed?
5. Helpfulness — does it actually address the user's real need?

Brief critique (2-4 sentences, specific):"""

REVISION_PROMPT_TEMPLATE = """\
Improve this response based on the critique.

Original question: {question}
Original response: {response}
Critique: {critique}

Write an improved response that addresses these issues:"""

# ── Text generation ───────────────────────────────────────────────────────────

def _generate_raw(prompt: str, max_new_tokens: int, temperature: float,
                  top_p: float, top_k: int, rep_penalty: float, seed: int) -> str:
    torch.manual_seed(int(seed))
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=880)
    gen_cfg = GenerationConfig(
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
        repetition_penalty=float(rep_penalty),
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    with torch.no_grad():
        out = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            generation_config=gen_cfg,
        )
    new_ids  = out[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    for stop in ["<|user|>", "<|end|>", "User:", "Human:", "\nUser", "\nHuman"]:
        if stop in response:
            response = response[:response.index(stop)].strip()
    return response

def get_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("text") or item.get("content") or "")
            else:
                parts.append(str(item))
        return " ".join(p for p in parts if p).strip()
    return str(content)

# ── Prompt builders ───────────────────────────────────────────────────────────

def build_prompt(message: str, history: list, four_stream: bool) -> str:
    lines = [f"<|system|>\n{ORB_SYSTEM}"]
    for entry in history:
        if not isinstance(entry, dict):
            continue
        role    = entry.get("role", "")
        content = get_text(entry.get("content", ""))
        if role == "user":
            lines.append(f"<|user|>\n{content}")
        elif role == "assistant":
            lines.append(f"<|orb|>\n{content}<|end|>")
    suffix = FOUR_STREAM_SUFFIX if four_stream else "\n"
    lines.append(f"<|user|>\n{message}{suffix}<|orb|>")
    return "\n".join(lines)

# ── Constitutional self-critique ──────────────────────────────────────────────

def self_critique(message: str, response: str, max_new_tokens: int,
                  temperature: float, top_p: float, top_k: int,
                  rep_penalty: float, seed: int) -> tuple[str, str]:
    """
    Phase 2 of Constitutional Compression (OASIS):
    1. Critique the initial response against the Orb constitution.
    2. Revise based on critique.
    Returns (revised_response, critique_text).
    """
    critique_prompt = CRITIQUE_PROMPT_TEMPLATE.format(
        question=message, response=response
    )
    critique = _generate_raw(
        critique_prompt,
        max_new_tokens=min(max_new_tokens, 140),
        temperature=max(temperature - 0.1, 0.5),
        top_p=top_p, top_k=top_k, rep_penalty=rep_penalty, seed=seed + 1,
    )

    revision_prompt = REVISION_PROMPT_TEMPLATE.format(
        question=message, response=response, critique=critique
    )
    revised = _generate_raw(
        revision_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p, top_k=top_k, rep_penalty=rep_penalty, seed=seed + 2,
    )

    return (revised if revised else response), critique

# ── Training status helper ────────────────────────────────────────────────────

def get_training_status() -> str:
    """Read last line of training log if running."""
    for log in ["logs/sft_training.log", "logs/star_loop.log"]:
        if os.path.isfile(log):
            try:
                with open(log, "rb") as f:
                    f.seek(-2048, 2)
                    last = f.read().decode("utf-8", errors="ignore").strip().split("\n")[-1]
                    if last:
                        return last[:80]
            except OSError:
                pass
    return "No training in progress"

# ── Chat logic ────────────────────────────────────────────────────────────────

def user_turn(message: str, history: list):
    if not message.strip():
        return "", history
    return "", history + [{"role": "user", "content": message}]

def bot_turn(history: list, max_new_tokens, temperature, top_p, top_k,
             rep_penalty, seed, four_stream, use_critique):
    if not history:
        return history, "", ""

    last = history[-1]
    last_role    = last.get("role", "") if isinstance(last, dict) else getattr(last, "role", "")
    last_content = last.get("content", "") if isinstance(last, dict) else getattr(last, "content", "")
    if last_role != "user":
        return history, "", ""

    message = get_text(last_content)
    past    = history[:-1]

    try:
        prompt   = build_prompt(message, past, four_stream)
        response = _generate_raw(prompt, max_new_tokens, temperature,
                                  top_p, top_k, rep_penalty, seed)

        if not response:
            response = "_(Orb returned an empty response — try rephrasing or lowering temperature.)_"

        critique_text = ""
        if use_critique:
            response, critique_text = self_critique(
                message, response, max_new_tokens, temperature,
                top_p, top_k, rep_penalty, seed
            )

        return history + [{"role": "assistant", "content": response}], critique_text, get_training_status()

    except gr.Error:
        raise
    except Exception as exc:
        print(f"[bot_turn error]\n{traceback.format_exc()}")
        raise gr.Error(f"Generation failed: {exc}") from exc

# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
body, .gradio-container { background: #080b14 !important; }

#page-header {
    background: linear-gradient(135deg, #0a0f1e 0%, #0f1a35 60%, #0a2240 100%);
    border-bottom: 1px solid #1a2844;
    padding: 18px 32px 14px;
    display: flex; align-items: center; gap: 16px;
}
#page-header .logo { font-size: 2.4rem; line-height: 1; }
#page-header .titles h1 {
    margin: 0; font-size: 1.5rem; font-weight: 700;
    background: linear-gradient(90deg, #7ec8e3, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em;
}
#page-header .titles p { margin: 3px 0 0; font-size: 0.78rem; color: #4a5568; }
#page-header .badge {
    margin-left: auto;
    background: #0f1d38; border: 1px solid #1e3a5f;
    border-radius: 20px; padding: 4px 14px;
    font-size: 0.72rem; color: #7ec8e3; font-family: monospace;
}

#chatbot {
    background: #0d1321 !important;
    border: 1px solid #1a2844 !important;
    border-radius: 14px !important;
    min-height: 480px;
}

#msg-box textarea {
    background: #0f1826 !important; border: 1px solid #1a2844 !important;
    color: #e2e8f0 !important; border-radius: 12px !important;
    font-size: 0.95rem; padding: 11px 15px !important; resize: none;
}
#msg-box textarea:focus { border-color: #7c3aed !important; outline: none !important; }
#msg-box textarea::placeholder { color: #2d3f5a !important; }

#send-btn {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    border: none !important; border-radius: 12px !important;
    color: #fff !important; font-weight: 600 !important;
    min-width: 90px; letter-spacing: 0.01em; transition: opacity 0.15s;
}
#send-btn:hover { opacity: 0.85 !important; }
#clear-btn {
    background: #111827 !important; border: 1px solid #1a2844 !important;
    border-radius: 12px !important; color: #4a5568 !important;
}

#settings-panel {
    background: #0d1321; border: 1px solid #1a2844;
    border-radius: 14px; padding: 18px 16px;
}
#settings-panel label { color: #94a3b8 !important; font-size: 0.82rem !important; }
#settings-panel input[type=range] { accent-color: #7c3aed !important; }
#settings-panel input[type=number] {
    background: #0f1826 !important; border: 1px solid #1a2844 !important;
    color: #e2e8f0 !important; border-radius: 8px !important;
}

#critique-box textarea {
    background: #0a1525 !important; border: 1px solid #1e3a5f !important;
    color: #7ec8e3 !important; font-size: 0.8rem !important;
    border-radius: 10px !important; font-family: monospace;
}
#status-box textarea {
    background: #0a120a !important; border: 1px solid #1a3a1a !important;
    color: #4ade80 !important; font-size: 0.75rem !important;
    border-radius: 8px !important; font-family: monospace;
}

.sh { color: #334155; font-size: 0.72rem; text-transform: uppercase;
      letter-spacing: 0.09em; font-weight: 600; margin-bottom: 8px; }

.chip-row { gap: 6px !important; flex-wrap: wrap; margin: 4px 0 8px !important; }
.chip-btn {
    background: #0f1826 !important; border: 1px solid #1a2844 !important;
    border-radius: 20px !important; padding: 5px 13px !important;
    font-size: 0.77rem !important; color: #64748b !important;
    cursor: pointer; transition: all 0.15s; height: auto !important;
    white-space: nowrap;
}
.chip-btn:hover {
    border-color: #7c3aed !important; color: #e2e8f0 !important;
    background: #1a1040 !important;
}

#status-bar {
    border-top: 1px solid #1a2844; padding: 10px 0 2px;
    display: flex; gap: 10px; justify-content: center; flex-wrap: wrap;
}
.pill {
    background: #0f1826; border: 1px solid #1a2844;
    border-radius: 20px; padding: 3px 12px;
    font-size: 0.71rem; color: #4a5568; font-family: monospace;
}
.pill span { color: #7ec8e3; }
.pill.green span { color: #4ade80; }
.pill.purple span { color: #a78bfa; }

.oasis-toggle label { color: #a78bfa !important; font-weight: 600 !important; }
"""

# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Orb — OASIS Model", css=CSS) as demo:

    # Header
    model_badge = MODEL_LABEL.split("(")[0].strip()
    gr.HTML(f"""
    <div id="page-header">
      <div class="logo">🔮</div>
      <div class="titles">
        <h1>Orb · OASIS Model</h1>
        <p>Orthogonal Autonomous Self-Improving System · Constitutional Reasoning · Four-Stream Analysis</p>
      </div>
      <div class="badge">{model_badge} · 117M params</div>
    </div>
    """)

    with gr.Row(equal_height=False, variant="panel"):

        # ── Chat column ──────────────────────────────────────────────────────
        with gr.Column(scale=3):

            chatbot = gr.Chatbot(
                label="",
                elem_id="chatbot",
                show_label=False,
                height=480,
                layout="bubble",
                avatar_images=(None, "🔮"),
                render_markdown=True,
                placeholder=(
                    "<div style='text-align:center;color:#1e3a5f;padding:60px 0'>"
                    "<div style='font-size:3rem;margin-bottom:12px'>🔮</div>"
                    "<div style='font-size:1.1rem;font-weight:600;color:#2d4a7a'>Orb is ready</div>"
                    "<div style='font-size:0.85rem;margin-top:6px;color:#1e3a5f'>"
                    "OASIS training framework active · Ask anything</div>"
                    "</div>"
                ),
            )

            with gr.Row(variant="panel"):
                msg = gr.Textbox(
                    placeholder="Ask Orb anything… constitutional reasoning and self-critique active",
                    show_label=False, lines=2, max_lines=6,
                    elem_id="msg-box", scale=5, container=False,
                )
                with gr.Column(scale=1, min_width=100):
                    send_btn  = gr.Button("Send ↑", variant="primary",   elem_id="send-btn")
                    clear_btn = gr.Button("Clear",  variant="secondary", elem_id="clear-btn")

            gr.HTML('<div class="sh" style="padding-top:8px">Quick prompts</div>')
            with gr.Row(elem_classes=["chip-row"]):
                chip1 = gr.Button("Why does ice float on water?",           elem_classes=["chip-btn"], size="sm")
                chip2 = gr.Button("Explain machine learning simply",         elem_classes=["chip-btn"], size="sm")
                chip3 = gr.Button("What causes inflation?",                  elem_classes=["chip-btn"], size="sm")
                chip4 = gr.Button("Are you conscious?",                      elem_classes=["chip-btn"], size="sm")
                chip5 = gr.Button("Once upon a time in a land far away",     elem_classes=["chip-btn"], size="sm")

            # Constitutional self-critique display
            critique_box = gr.Textbox(
                label="🔍 Constitutional Self-Critique (OASIS Phase 2)",
                elem_id="critique-box",
                lines=4, interactive=False, visible=False,
                placeholder="Critique will appear here when Constitutional Mode is enabled…",
            )

            # Training status
            status_box = gr.Textbox(
                label="⚙ Training Status",
                elem_id="status-box",
                lines=1, interactive=False, visible=False,
                placeholder="Training log tail…",
            )

        # ── Settings sidebar ─────────────────────────────────────────────────
        with gr.Column(scale=1, elem_id="settings-panel"):
            gr.HTML('<div class="sh" style="margin-bottom:12px">🔮 OASIS Settings</div>')

            four_stream = gr.Checkbox(
                value=False,
                label="⊞ Four-Stream Reasoning (OASIS Phase 1)",
                info="Forces ANALYTICAL / SKEPTICAL / CONCRETE / SYNTHESIS reasoning",
                elem_classes=["oasis-toggle"],
            )
            use_critique = gr.Checkbox(
                value=False,
                label="⚖ Constitutional Self-Critique (OASIS Phase 2)",
                info="Orb critiques and revises its own response before showing it",
                elem_classes=["oasis-toggle"],
            )
            show_critique = gr.Checkbox(
                value=False, label="Show critique panel",
            )
            show_status = gr.Checkbox(
                value=False, label="Show training status",
            )

            gr.HTML('<hr style="border-color:#1a2844;margin:12px 0">')
            gr.HTML('<div class="sh">⚙ Generation settings</div>')

            max_new_tokens = gr.Slider(
                20, 400, value=200, step=10, label="Max new tokens",
            )
            temperature = gr.Slider(
                0.1, 2.0, value=0.85, step=0.05, label="Temperature",
                info="Higher → more creative",
            )
            top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.01, label="Top-p")
            top_k = gr.Slider(0, 100, value=50, step=1, label="Top-k")
            rep_penalty = gr.Slider(
                1.0, 2.0, value=1.1, step=0.05, label="Repetition penalty",
            )
            seed = gr.Number(value=42, label="Random seed", precision=0)

            gr.HTML(f"""
            <div style="margin-top:16px;border-top:1px solid #1a2844;padding-top:14px">
              <div class="sh">Model info</div>
              <div style="display:flex;flex-wrap:wrap;gap:5px">
                <span class="pill purple">model <span>Orb</span></span>
                <span class="pill">params <span>117 M</span></span>
                <span class="pill">ctx <span>1024 tok</span></span>
                <span class="pill">device <span>CPU</span></span>
                <span class="pill green">checkpoint <span>{model_badge}</span></span>
              </div>
            </div>
            """)

    gr.HTML("""
    <div id="status-bar">
      <span class="pill">framework <span>OASIS · HuggingFace</span></span>
      <span class="pill">phases <span>SFT → STaR → GGUF</span></span>
      <span class="pill purple">reasoning <span>4-stream + critique</span></span>
      <span class="pill">weights <span>local · models/</span></span>
    </div>
    """)

    # ── Event wiring ──────────────────────────────────────────────────────────
    settings = [max_new_tokens, temperature, top_p, top_k, rep_penalty, seed,
                four_stream, use_critique]

    def _send(message, history, *args):
        return user_turn(message, history)

    def _bot(history, *args):
        return bot_turn(history, *args)

    send_btn.click(
        fn=user_turn, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False,
    ).then(
        fn=bot_turn, inputs=[chatbot] + settings,
        outputs=[chatbot, critique_box, status_box],
    )

    msg.submit(
        fn=user_turn, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False,
    ).then(
        fn=bot_turn, inputs=[chatbot] + settings,
        outputs=[chatbot, critique_box, status_box],
    )

    clear_btn.click(fn=lambda: ([], "", ""), outputs=[chatbot, critique_box, status_box], queue=False)

    # Show/hide panels
    show_critique.change(fn=lambda v: gr.update(visible=v), inputs=show_critique, outputs=critique_box)
    show_status.change(fn=lambda v: gr.update(visible=v), inputs=show_status, outputs=status_box)

    # Chip buttons
    chips = [
        (chip1, "Why does ice float on water?"),
        (chip2, "Explain machine learning simply"),
        (chip3, "What causes inflation?"),
        (chip4, "Are you conscious?"),
        (chip5, "Once upon a time in a land far away"),
    ]
    for chip, text in chips:
        chip.click(fn=lambda t=text: t, outputs=msg, queue=False)

# ── Launch ────────────────────────────────────────────────────────────────────

PORT = int(os.environ.get("PORT", 8000))
demo.launch(
    server_name="0.0.0.0",
    server_port=PORT,
    show_error=True,
    css=CSS,
)
