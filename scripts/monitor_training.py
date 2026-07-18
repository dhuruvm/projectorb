"""
Orb Training Monitor — OASIS Pipeline
Shows live tail of training logs, current loss, adapter status, and ETA.

Usage:
    python scripts/monitor_training.py           # watch all logs
    python scripts/monitor_training.py --once    # print once and exit
"""

import argparse
import os
import re
import sys
import time

LOG_FILES = {
    "SFT LoRA":   "logs/sft_training.log",
    "STaR Loop":  "logs/star_loop.log",
    "Merge":      "logs/merge.log",
}

ADAPTERS = {
    "Orb-STaR":    "models/orb-star/adapter_config.json",
    "Orb-SFT":     "models/orb-sft-lora/adapter_config.json",
    "Orb-Merged":  "models/orb-merged/config.json",
    "GGUF F16":    "models/orb-f16.gguf",
    "GGUF Q4_K_M": "models/orb-Q4_K_M.gguf",
}

OASIS_PHASES = [
    ("Phase 0", "SFT LoRA",             "models/orb-sft-lora/adapter_config.json"),
    ("Phase 1", "Orthogonal Reasoning", None),
    ("Phase 2", "Constitutional DPO",   None),
    ("Phase 3", "SPIN",                 None),
    ("Phase 4", "STaR Self-Improvement","models/orb-star/adapter_config.json"),
    ("Phase 5", "Merge + GGUF Export",  "models/orb-merged/config.json"),
]


def _size_str(path: str) -> str:
    try:
        b = os.path.getsize(path)
        for unit in ("B", "KB", "MB", "GB"):
            if b < 1024:
                return f"{b:.1f} {unit}"
            b /= 1024
        return f"{b:.1f} TB"
    except OSError:
        return "?"


def _tail(path: str, n_bytes: int = 4096) -> list[str]:
    """Return last n_bytes of a file as a list of stripped lines."""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - n_bytes))
            raw = f.read().decode("utf-8", errors="ignore")
        return [l.strip() for l in raw.splitlines() if l.strip()]
    except OSError:
        return []


def _parse_loss(lines: list[str]) -> str:
    """Extract the latest loss value from HuggingFace Trainer log lines."""
    loss_pat = re.compile(r"'loss':\s*([\d.]+)")
    eval_pat  = re.compile(r"'eval_loss':\s*([\d.]+)")
    train_loss = eval_loss = None
    for line in reversed(lines):
        if eval_loss is None:
            m = eval_pat.search(line)
            if m:
                eval_loss = m.group(1)
        if train_loss is None:
            m = loss_pat.search(line)
            if m:
                train_loss = m.group(1)
        if train_loss and eval_loss:
            break
    parts = []
    if train_loss:
        parts.append(f"train_loss={train_loss}")
    if eval_loss:
        parts.append(f"eval_loss={eval_loss}")
    return "  ".join(parts) if parts else "—"


def _parse_step(lines: list[str]) -> str:
    """Extract current step / total steps."""
    step_pat = re.compile(r"\[(\d+)/(\d+)\]|Step\s+(\d+)")
    for line in reversed(lines):
        m = step_pat.search(line)
        if m:
            if m.group(1):
                return f"step {m.group(1)}/{m.group(2)}"
            return f"step {m.group(3)}"
    return "—"


def _active_log() -> tuple[str, str] | tuple[None, None]:
    """Return (label, path) of the most-recently modified log, if any."""
    best = (0, None, None)
    for label, path in LOG_FILES.items():
        if os.path.isfile(path):
            mtime = os.path.getmtime(path)
            if mtime > best[0]:
                best = (mtime, label, path)
    if best[1]:
        return best[1], best[2]
    return None, None


def _separator(char="─", width=60) -> str:
    return char * width


def print_status():
    print(_separator("═"))
    print("  🔮  Orb Training Monitor — OASIS Pipeline")
    print(_separator("═"))

    # OASIS phase progress
    print("\n  OASIS Phases:")
    for phase, name, sentinel in OASIS_PHASES:
        if sentinel is None:
            status = "⬜  planned"
        elif os.path.isfile(sentinel):
            status = f"✅  complete  ({_size_str(sentinel)})"
        else:
            # Check if this is the currently running phase
            active_label, _ = _active_log()
            running = active_label and phase_matches(phase, active_label)
            status = "🔄  running…" if running else "⬜  not started"
        print(f"    {phase}: {name:<28} {status}")

    # Adapter / model artifact inventory
    print(f"\n  Model Artifacts:")
    for name, path in ADAPTERS.items():
        exists = os.path.isfile(path)
        marker = "✅" if exists else "⬜"
        size   = f"  ({_size_str(path)})" if exists else ""
        print(f"    {marker}  {name:<18} {path}{size}")

    # Active log tail
    label, log_path = _active_log()
    if label and log_path:
        lines = _tail(log_path)
        loss  = _parse_loss(lines)
        step  = _parse_step(lines)
        last  = lines[-1][:100] if lines else "—"
        print(f"\n  Active Training: {label}")
        print(f"    {step:<20}  loss: {loss}")
        print(f"    Last line: {last}")
    else:
        print("\n  No training log detected. Start with:")
        print("    nohup python scripts/train_sft_lora.py > logs/sft_training.log 2>&1 &")

    print(f"\n{_separator()}\n")


def phase_matches(phase: str, label: str) -> bool:
    return (phase == "Phase 0" and "SFT" in label) or \
           (phase == "Phase 4" and "STaR" in label)


def main():
    parser = argparse.ArgumentParser(description="Orb Training Monitor")
    parser.add_argument("--once",     action="store_true", help="Print once and exit")
    parser.add_argument("--interval", type=int, default=30, help="Refresh interval in seconds (default: 30)")
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)

    if args.once:
        print_status()
        return

    print(f"Refreshing every {args.interval}s — Ctrl+C to stop\n")
    try:
        while True:
            # Clear terminal
            os.system("clear" if os.name != "nt" else "cls")
            print_status()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")


if __name__ == "__main__":
    main()
