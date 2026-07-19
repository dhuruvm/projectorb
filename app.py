"""
Obscuro — OASIS Cognitive Agent · Gradio 6 interface
Created by The Director.

Thin UI layer wrapping OrbAgent. All reasoning logic lives in orb/.
"""
from __future__ import annotations

import os
import traceback

import gradio as gr

from orb.agent import OrbAgent, AgentOptions

# ── Agent singleton (shared across all sessions) ──────────────────────────────

agent = OrbAgent()

# ── Helpers ───────────────────────────────────────────────────────────────────

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

# ── Chat handlers ─────────────────────────────────────────────────────────────

def user_turn(message: str, history: list):
    if not message.strip():
        return "", history
    return "", history + [{"role": "user", "content": message}]


def bot_turn(
    history: list,
    max_new_tokens, temperature, top_p, top_k, rep_penalty, seed,
    four_stream, use_critique, multi_path,
):
    if not history:
        return history, "", "", ""

    last = history[-1]
    if not isinstance(last, dict) or last.get("role") != "user":
        return history, "", "", ""

    message = _get_text(last.get("content", ""))
    past    = history[:-1]

    try:
        opts = AgentOptions(
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            repetition_penalty=float(rep_penalty),
            seed=int(seed),
            four_stream=bool(four_stream),
            use_critique=bool(use_critique),
            multi_path=bool(multi_path),
        )
        result = agent.run(message, past, opts)

        stats      = agent.memory_stats()
        status_str = (
            f"Memory — episodes: {stats['episodes']} | lessons: {stats['lessons']} | "
            f"time: {result.elapsed_ms} ms"
        )
        paths_str = ""
        if result.reasoning_paths:
            paths_str = "  ".join(
                f"T={p.temperature:.2f}→{p.combined_score:.3f}"
                for p in result.reasoning_paths
            )

        return (
            history + [{"role": "assistant", "content": result.response}],
            result.critique,
            status_str,
            paths_str,
        )

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
    min-height: 460px;
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
    min-width: 90px; transition: opacity 0.15s;
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
#paths-box textarea {
    background: #0f0a1a !important; border: 1px solid #2a1a4a !important;
    color: #a78bfa !important; font-size: 0.75rem !important;
    border-radius: 8px !important; font-family: monospace;
}

.sh { color: #334155; font-size: 0.72rem; text-transform: uppercase;
      letter-spacing: 0.09em; font-weight: 600; margin-bottom: 8px; }

.chip-btn {
    background: #0f1826 !important; border: 1px solid #1a2844 !important;
    border-radius: 20px !important; padding: 5px 13px !important;
    font-size: 0.77rem !important; color: #64748b !important;
    cursor: pointer; transition: all 0.15s; height: auto !important;
}
.chip-btn:hover {
    border-color: #7c3aed !important; color: #e2e8f0 !important;
    background: #1a1040 !important;
}

.pill {
    background: #0f1826; border: 1px solid #1a2844;
    border-radius: 20px; padding: 3px 12px;
    font-size: 0.71rem; color: #4a5568; font-family: monospace;
}
.pill span { color: #7ec8e3; }
.pill.green span { color: #4ade80; }
.pill.purple span { color: #a78bfa; }
"""

# ── UI ────────────────────────────────────────────────────────────────────────

model_badge = agent.model_label.split("(")[0].strip()

with gr.Blocks(title="Obscuro — OASIS Cognitive Agent") as demo:

    gr.HTML(f"""
    <div id="page-header">
      <div class="logo">🔮</div>
      <div class="titles">
        <h1>Obscuro · OASIS Cognitive Agent</h1>
        <p>Created by The Director · Multi-path Reasoning · Constitutional Critique · Persistent Memory</p>
      </div>
      <div class="badge">{model_badge} · 1.24B params</div>
    </div>
    """)

    with gr.Row(equal_height=False, variant="panel"):

        # ── Chat column ──────────────────────────────────────────────────────
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="", elem_id="chatbot", show_label=False,
                height=460, layout="bubble",
                avatar_images=(None, "🔮"), render_markdown=True,
                placeholder=(
                    "<div style='text-align:center;color:#1e3a5f;padding:60px 0'>"
                    "<div style='font-size:3rem;margin-bottom:12px'>🔮</div>"
                    "<div style='font-size:1.1rem;font-weight:600;color:#2d4a7a'>Obscuro is ready</div>"
                    "<div style='font-size:0.85rem;margin-top:6px;color:#1e3a5f'>"
                    "Created by The Director · Cognitive loop active</div>"
                    "</div>"
                ),
            )

            with gr.Row(variant="panel"):
                msg = gr.Textbox(
                    placeholder="Ask Obscuro anything…",
                    show_label=False, lines=2, max_lines=6,
                    elem_id="msg-box", scale=5, container=False,
                )
                with gr.Column(scale=1, min_width=100):
                    send_btn  = gr.Button("Send ↑", variant="primary",   elem_id="send-btn")
                    clear_btn = gr.Button("Clear",  variant="secondary", elem_id="clear-btn")

            gr.HTML('<div class="sh" style="padding-top:8px">Quick prompts</div>')
            with gr.Row():
                chip1 = gr.Button("Why does ice float on water?",    elem_classes=["chip-btn"], size="sm")
                chip2 = gr.Button("Explain machine learning simply",  elem_classes=["chip-btn"], size="sm")
                chip3 = gr.Button("What causes inflation?",           elem_classes=["chip-btn"], size="sm")
                chip4 = gr.Button("Are you conscious?",               elem_classes=["chip-btn"], size="sm")

            critique_box = gr.Textbox(
                label="🔍 Constitutional Self-Critique",
                elem_id="critique-box", lines=3, interactive=False, visible=False,
                placeholder="Critique appears here when Constitutional Mode is enabled…",
            )
            status_box = gr.Textbox(
                label="⚙ Memory & Timing",
                elem_id="status-box", lines=1, interactive=False, visible=False,
            )
            paths_box = gr.Textbox(
                label="🧠 Reasoning Path Scores (temp → combined score)",
                elem_id="paths-box", lines=1, interactive=False, visible=False,
            )

        # ── Settings sidebar ─────────────────────────────────────────────────
        with gr.Column(scale=1, elem_id="settings-panel"):
            gr.HTML('<div class="sh" style="margin-bottom:12px">🔮 OASIS Settings</div>')

            multi_path   = gr.Checkbox(value=True,  label="🧠 Multi-Path Reasoning  (3 candidates)")
            four_stream  = gr.Checkbox(value=False,  label="⊞ Four-Stream Reasoning")
            use_critique = gr.Checkbox(value=False, label="⚖ Constitutional Self-Critique")

            gr.HTML('<div class="sh" style="margin-top:12px">Inspect</div>')
            show_critique = gr.Checkbox(value=False, label="Show critique panel")
            show_status   = gr.Checkbox(value=False, label="Show memory & timing")
            show_paths    = gr.Checkbox(value=False, label="Show reasoning path scores")

            gr.HTML('<hr style="border-color:#1a2844;margin:12px 0">')
            gr.HTML('<div class="sh">⚙ Generation settings</div>')

            max_new_tokens = gr.Slider(20,  400, value=200, step=10,   label="Max new tokens")
            temperature    = gr.Slider(0.1, 2.0, value=0.85, step=0.05, label="Temperature")
            top_p          = gr.Slider(0.1, 1.0, value=0.95, step=0.01, label="Top-p")
            top_k          = gr.Slider(0,   100, value=50,   step=1,    label="Top-k")
            rep_penalty    = gr.Slider(1.0, 2.0, value=1.1,  step=0.05, label="Repetition penalty")
            seed           = gr.Number(value=42, label="Random seed", precision=0)

            gr.HTML(f"""
            <div style="margin-top:16px;border-top:1px solid #1a2844;padding-top:14px">
              <div class="sh">Model info</div>
              <div style="display:flex;flex-wrap:wrap;gap:5px">
                <span class="pill purple">model <span>Obscuro</span></span>
                <span class="pill">by <span>The Director</span></span>
                <span class="pill">params <span>117 M</span></span>
                <span class="pill">ctx <span>1024 tok</span></span>
                <span class="pill green">checkpoint <span>{model_badge}</span></span>
              </div>
            </div>
            """)

    gr.HTML("""
    <div style="border-top:1px solid #1a2844;padding:10px 0 2px;
                display:flex;gap:10px;justify-content:center;flex-wrap:wrap;">
      <span class="pill purple">Obscuro <span>by The Director</span></span>
      <span class="pill">framework <span>OASIS · HuggingFace</span></span>
      <span class="pill">architecture <span>multi-path + critique + memory</span></span>
    </div>
    """)

    # ── Event wiring ──────────────────────────────────────────────────────────

    settings = [
        max_new_tokens, temperature, top_p, top_k, rep_penalty, seed,
        four_stream, use_critique, multi_path,
    ]

    send_btn.click(
        fn=user_turn, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False,
    ).then(
        fn=bot_turn, inputs=[chatbot] + settings,
        outputs=[chatbot, critique_box, status_box, paths_box],
    )

    msg.submit(
        fn=user_turn, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False,
    ).then(
        fn=bot_turn, inputs=[chatbot] + settings,
        outputs=[chatbot, critique_box, status_box, paths_box],
    )

    clear_btn.click(
        fn=lambda: ([], "", "", ""),
        outputs=[chatbot, critique_box, status_box, paths_box],
        queue=False,
    )

    # Panel visibility toggles
    show_critique.change(fn=lambda v: gr.update(visible=v), inputs=show_critique, outputs=critique_box)
    show_status.change(  fn=lambda v: gr.update(visible=v), inputs=show_status,   outputs=status_box)
    show_paths.change(   fn=lambda v: gr.update(visible=v), inputs=show_paths,    outputs=paths_box)

    # Chip buttons
    for chip, text in [
        (chip1, "Why does ice float on water?"),
        (chip2, "Explain machine learning simply"),
        (chip3, "What causes inflation?"),
        (chip4, "Are you conscious?"),
    ]:
        chip.click(fn=lambda t=text: t, outputs=msg, queue=False)

# ── Launch ────────────────────────────────────────────────────────────────────

PORT = int(os.environ.get("PORT", 8000))
demo.launch(
    server_name="0.0.0.0",
    server_port=PORT,
    show_error=True,
    css=CSS,
)
