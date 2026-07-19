"""
Convert merged Orb model to GGUF format for llama.cpp.
Produces both an F16 GGUF and a Q4_K_M quantized GGUF.

Usage:
    python scripts/convert_to_gguf.py

Requirements:
    - models/orb-merged/ must exist (run merge_lora.py first)
    - Builds llama.cpp from source if not present (~5 minutes)

Output:
    models/orb-f16.gguf      (~225MB, full quality)
    models/orb-Q4_K_M.gguf  (~70MB,  recommended for deployment)
"""

import os
import subprocess
import sys
from pathlib import Path


LLAMA_CPP_DIR = Path("llama.cpp")
MERGED_MODEL_DIR = Path("models/orb-merged")
GGUF_F16 = Path("models/orb-f16.gguf")
GGUF_Q4 = Path("models/orb-Q4_K_M.gguf")


def ensure_merged_model():
    if not MERGED_MODEL_DIR.exists():
        print("Merged model not found. Run first:")
        print("  python scripts/merge_lora.py")
        sys.exit(1)
    print(f"✓ Found merged model at {MERGED_MODEL_DIR}")


def clone_llama_cpp():
    if LLAMA_CPP_DIR.exists():
        print(f"✓ llama.cpp already cloned at {LLAMA_CPP_DIR}")
        return

    print("Cloning llama.cpp (shallow clone)...")
    result = subprocess.run(
        ["git", "clone", "--depth=1",
         "https://github.com/ggerganov/llama.cpp",
         str(LLAMA_CPP_DIR)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Clone failed:\n{result.stderr}")
        sys.exit(1)
    print("✓ llama.cpp cloned")


def install_conversion_deps():
    print("Installing GGUF conversion dependencies...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "gguf", "sentencepiece", "-q"],
        check=True
    )
    print("✓ Dependencies installed")


def build_llama_cpp():
    quantize_bin = LLAMA_CPP_DIR / "llama-quantize"
    if quantize_bin.exists():
        print("✓ llama.cpp already built")
        return

    print("Building llama.cpp (CPU-only build)...")
    print("  This takes 2–5 minutes...")

    n_cores = os.cpu_count() or 2
    result = subprocess.run(
        ["make", f"-j{n_cores}", "-C", str(LLAMA_CPP_DIR)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        # Try cmake as fallback
        print("  make failed, trying cmake...")
        build_dir = LLAMA_CPP_DIR / "build"
        build_dir.mkdir(exist_ok=True)
        subprocess.run(
            ["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"],
            cwd=str(build_dir), check=True
        )
        subprocess.run(
            ["cmake", "--build", ".", f"-j{n_cores}"],
            cwd=str(build_dir), check=True
        )
    print("✓ llama.cpp built")


def convert_to_f16():
    if GGUF_F16.exists():
        print(f"✓ F16 GGUF already exists at {GGUF_F16}")
        return

    print(f"\nConverting to F16 GGUF...")
    convert_script = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"

    if not convert_script.exists():
        # Try alternate path
        convert_script = LLAMA_CPP_DIR / "convert-hf-to-gguf.py"

    if not convert_script.exists():
        print(f"Conversion script not found in {LLAMA_CPP_DIR}")
        print("Try: ls llama.cpp/*.py")
        sys.exit(1)

    result = subprocess.run(
        [
            sys.executable,
            str(convert_script),
            str(MERGED_MODEL_DIR),
            "--outfile", str(GGUF_F16),
            "--outtype", "f16",
            "--vocab-type", "bpe",
        ],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        print(f"Conversion failed:\n{result.stderr}\n{result.stdout}")
        sys.exit(1)

    size_mb = GGUF_F16.stat().st_size / 1024 / 1024
    print(f"✓ F16 GGUF created: {GGUF_F16} ({size_mb:.1f} MB)")


def quantize_to_q4():
    if GGUF_Q4.exists():
        print(f"✓ Q4_K_M GGUF already exists at {GGUF_Q4}")
        return

    # Find quantize binary
    quantize_bin = LLAMA_CPP_DIR / "llama-quantize"
    if not quantize_bin.exists():
        quantize_bin = LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize"
    if not quantize_bin.exists():
        print("llama-quantize binary not found. Skipping quantization.")
        print("The F16 GGUF is still usable with llama.cpp.")
        return

    print(f"\nQuantizing to Q4_K_M...")
    result = subprocess.run(
        [str(quantize_bin), str(GGUF_F16), str(GGUF_Q4), "Q4_K_M"],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        print(f"Quantization failed:\n{result.stderr}")
        return

    f16_mb = GGUF_F16.stat().st_size / 1024 / 1024
    q4_mb = GGUF_Q4.stat().st_size / 1024 / 1024
    compression = (1 - q4_mb / f16_mb) * 100
    print(f"✓ Q4_K_M GGUF created: {GGUF_Q4} ({q4_mb:.1f} MB)")
    print(f"  Compression: {f16_mb:.1f} MB → {q4_mb:.1f} MB ({compression:.0f}% reduction)")


def print_usage():
    print("\n" + "═" * 60)
    print("  Orb GGUF files ready!")
    print("═" * 60)

    for path, label in [(GGUF_F16, "F16 (full quality)"), (GGUF_Q4, "Q4_K_M (recommended)")]:
        if path.exists():
            mb = path.stat().st_size / 1024 / 1024
            print(f"  {path}  [{mb:.1f} MB]  — {label}")

    print("""
Run Orb with llama.cpp:
  ./llama.cpp/llama-cli \\
      -m models/orb-Q4_K_M.gguf \\
      -p "<|system|>\\nYou are Orb.\\n<|user|>\\nHello!\\n<|orb|>\\n" \\
      -n 200 --temp 0.7

Start Orb as OpenAI-compatible server:
  ./llama.cpp/llama-server \\
      -m models/orb-Q4_K_M.gguf \\
      --port 8080 --host 0.0.0.0

Use with Ollama (once you have Ollama installed):
  ollama create orb -f Modelfile
  ollama run orb
""")


def main():
    ensure_merged_model()
    clone_llama_cpp()
    install_conversion_deps()
    build_llama_cpp()
    convert_to_f16()
    quantize_to_q4()
    print_usage()


if __name__ == "__main__":
    main()
