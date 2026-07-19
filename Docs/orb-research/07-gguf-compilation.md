# 07 — GGUF Compilation: Converting Orb to llama.cpp

## What Is GGUF?

**GGUF** (GPT-Generated Unified Format) is the file format used by **llama.cpp** — the C++ inference engine that runs LLMs efficiently on CPU without GPU. It is the standard format for local LLM deployment in 2024-2025.

**Why GGUF / llama.cpp?**
- Runs at 2–10× the speed of PyTorch on CPU
- Dramatically lower memory usage through quantization
- No Python required at inference time — pure C++
- Portable: same file runs on Mac, Linux, Windows, Raspberry Pi
- Supports quantization from 2-bit to 16-bit (Q2_K to F16)
- The format used by Ollama, LM Studio, Jan.ai, and most local LLM tools

---

## Quantization Options Explained

| Format | Bits/Weight | Size (117M model) | Quality Loss | Speed |
|--------|-------------|-------------------|--------------|-------|
| F32 | 32-bit | ~450MB | None | Baseline |
| F16 | 16-bit | ~225MB | Negligible | 1.2× faster |
| Q8_0 | 8-bit | ~120MB | Tiny | 1.5× faster |
| **Q4_K_M** | **~4.5-bit** | **~70MB** | Small | **2–3× faster** |
| Q4_K_S | ~4-bit | ~65MB | Small | 2–3× faster |
| Q3_K_M | ~3.5-bit | ~52MB | Moderate | 3× faster |
| Q2_K | ~2.4-bit | ~38MB | Significant | 4× faster |

**Recommended for Orb: Q4_K_M**
- Best quality-to-speed tradeoff
- 70MB file — easily shareable
- Quality loss is < 1% on most benchmarks vs F16
- Runs at ~20–40 tokens/second on a modern CPU

---

## Step 1: Install llama.cpp

```bash
# Clone and build llama.cpp from source
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Build (CPU only, no GPU)
make -j$(nproc)

# Verify build
./llama-cli --version
```

**On Replit (NixOS):**
```bash
# Ensure build tools are available
nix-env -iA nixpkgs.cmake nixpkgs.gcc

# Or use the pre-built approach with Python binding
pip install llama-cpp-python
```

---

## Step 2: Merge LoRA Adapter into Base Model

Before converting to GGUF, merge the LoRA adapter weights back into the base model to get a single, standalone set of weights.

```python
# scripts/merge_lora.py
"""
Merge LoRA adapter into base model weights.
This produces a single model directory that can be converted to GGUF.
"""

import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import PeftModel

def merge_lora_into_base(
    base_model_path: str = "models/gpt2",
    lora_adapter_path: str = "models/orb-star",   # or orb-sft-lora
    output_path: str = "models/orb-merged"
):
    print("Loading base model...")
    tokenizer = GPT2Tokenizer.from_pretrained(lora_adapter_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = GPT2LMHeadModel.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16   # Save memory during merge
    )
    base_model.resize_token_embeddings(len(tokenizer))
    
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    
    print("Merging LoRA weights into base model (this is permanent)...")
    model = model.merge_and_unload()
    
    print(f"Saving merged model to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    
    # Verify
    size_mb = sum(
        os.path.getsize(os.path.join(output_path, f))
        for f in os.listdir(output_path)
        if f.endswith(".safetensors") or f.endswith(".bin")
    ) / 1024 / 1024
    
    print(f"\nMerge complete!")
    print(f"  Output: {output_path}")
    print(f"  Model size: {size_mb:.1f} MB")
    print(f"  Ready for GGUF conversion")


if __name__ == "__main__":
    merge_lora_into_base()
```

---

## Step 3: Convert to GGUF Format

**IMPORTANT: llama.cpp's conversion scripts are built for LLaMA-style models. GPT-2 requires special handling.**

### Option A: Use the GPT-2 conversion script (direct)

```bash
# llama.cpp includes a conversion script for GPT-2
cd llama.cpp

python convert_hf_to_gguf.py \
    ../models/orb-merged \
    --outfile ../models/orb.gguf \
    --outtype f16

# Verify the conversion
./llama-cli -m ../models/orb.gguf -p "Hello, I am Orb" -n 50
```

### Option B: Use Python gguf library (alternative)

```python
# scripts/convert_to_gguf.py
"""
Convert merged Orb model to GGUF format using the gguf Python library.
This is the pure-Python approach that doesn't require building llama.cpp.
"""

import struct
import numpy as np
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

def convert_gpt2_to_gguf(model_path: str, output_path: str, quantize: str = "f16"):
    """
    Convert a GPT-2 (or fine-tuned GPT-2) model to GGUF format.
    
    quantize options:
      - "f32": 32-bit float (largest, best quality)
      - "f16": 16-bit float (recommended starting point)
      - For Q4_K_M etc., convert to f16 first, then quantize with llama.cpp CLI
    """
    
    # The official approach: use llama.cpp's convert script
    import subprocess
    import sys
    
    # Check if llama.cpp is available
    llama_cpp_path = Path("llama.cpp")
    if not llama_cpp_path.exists():
        print("Cloning llama.cpp...")
        subprocess.run(["git", "clone", "--depth=1",
                       "https://github.com/ggerganov/llama.cpp",
                       "llama.cpp"], check=True)
    
    # Install conversion dependencies
    subprocess.run([sys.executable, "-m", "pip", "install", "gguf", "sentencepiece"],
                  check=True)
    
    print(f"Converting {model_path} to GGUF ({quantize})...")
    
    result = subprocess.run([
        sys.executable,
        "llama.cpp/convert_hf_to_gguf.py",
        model_path,
        "--outfile", output_path,
        "--outtype", quantize,
        "--vocab-type", "bpe"       # GPT-2 uses BPE tokenization
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Conversion failed:\n{result.stderr}")
        raise RuntimeError("GGUF conversion failed")
    
    size_mb = Path(output_path).stat().st_size / 1024 / 1024
    print(f"\nConversion successful!")
    print(f"  Output: {output_path}")
    print(f"  Size: {size_mb:.1f} MB")
    
    return output_path


def quantize_gguf(input_gguf: str, output_gguf: str, quantization: str = "Q4_K_M"):
    """
    Apply quantization to an F16 GGUF file using llama.cpp's quantize tool.
    Must build llama.cpp first: cd llama.cpp && make
    """
    import subprocess
    
    quantize_bin = "llama.cpp/llama-quantize"
    
    result = subprocess.run([
        quantize_bin,
        input_gguf,
        output_gguf,
        quantization
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Quantization failed:\n{result.stderr}")
        raise RuntimeError("Quantization failed")
    
    original_mb = Path(input_gguf).stat().st_size / 1024 / 1024
    quantized_mb = Path(output_gguf).stat().st_size / 1024 / 1024
    compression = (1 - quantized_mb / original_mb) * 100
    
    print(f"\nQuantization complete!")
    print(f"  Format: {quantization}")
    print(f"  Original (F16): {original_mb:.1f} MB")
    print(f"  Quantized:      {quantized_mb:.1f} MB")
    print(f"  Compression:    {compression:.1f}%")


if __name__ == "__main__":
    # Step 1: Convert to F16 GGUF
    convert_gpt2_to_gguf(
        model_path="models/orb-merged",
        output_path="models/orb-f16.gguf",
        quantize="f16"
    )
    
    # Step 2: Build llama.cpp (needed for quantization)
    import subprocess
    subprocess.run(["make", "-C", "llama.cpp", "-j4"], check=True)
    
    # Step 3: Quantize to Q4_K_M (recommended)
    quantize_gguf(
        input_gguf="models/orb-f16.gguf",
        output_gguf="models/orb-Q4_K_M.gguf",
        quantization="Q4_K_M"
    )
    
    print("\nOrb GGUF files ready:")
    print("  models/orb-f16.gguf      (F16, full quality)")
    print("  models/orb-Q4_K_M.gguf   (Q4_K_M, recommended for deployment)")
```

---

## Step 4: Run Orb with llama.cpp

After conversion, Orb runs entirely in C++ — no Python, no PyTorch, no GPU needed.

```bash
# Basic inference
./llama.cpp/llama-cli \
    -m models/orb-Q4_K_M.gguf \
    -p "<|system|>\nYou are Orb, an advanced AI.\n<|user|>\nWhat is the capital of France?\n<|orb|>\n" \
    -n 200 \
    --temp 0.7 \
    --top-p 0.95 \
    --repeat-penalty 1.1

# Interactive chat mode
./llama.cpp/llama-cli \
    -m models/orb-Q4_K_M.gguf \
    --interactive \
    --in-prefix "<|user|>\n" \
    --in-suffix "\n<|orb|>\n" \
    --reverse-prompt "<|user|>" \
    -n 512

# Server mode (OpenAI-compatible API)
./llama.cpp/llama-server \
    -m models/orb-Q4_K_M.gguf \
    --port 8080 \
    --host 0.0.0.0 \
    --ctx-size 2048

# Then call it like OpenAI API:
# curl http://localhost:8080/v1/chat/completions \
#   -H "Content-Type: application/json" \
#   -d '{"messages": [{"role": "user", "content": "Hello Orb"}]}'
```

---

## Step 5: llama.cpp Server as Gradio Backend

Replace the PyTorch inference in `app.py` with llama.cpp server calls — dramatically faster on CPU:

```python
# Alternative app.py backend using llama.cpp server

import requests
import subprocess
import os
import time

LLAMA_SERVER_PORT = 8081
LLAMA_SERVER_URL = f"http://localhost:{LLAMA_SERVER_PORT}"
GGUF_PATH = "models/orb-Q4_K_M.gguf"

def start_llama_server():
    """Start llama.cpp server as a background process."""
    if not os.path.exists(GGUF_PATH):
        print(f"GGUF model not found at {GGUF_PATH}. Using PyTorch fallback.")
        return None
    
    proc = subprocess.Popen([
        "llama.cpp/llama-server",
        "-m", GGUF_PATH,
        "--port", str(LLAMA_SERVER_PORT),
        "--host", "127.0.0.1",
        "--ctx-size", "2048",
        "--threads", str(os.cpu_count() or 4),
        "--n-predict", "400",
        "--temp", "0.7",
        "-ngl", "0",          # 0 GPU layers — CPU only
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Wait for server to be ready
    for _ in range(30):
        try:
            requests.get(f"{LLAMA_SERVER_URL}/health", timeout=1)
            print(f"[startup] llama.cpp server running on port {LLAMA_SERVER_PORT}")
            return proc
        except:
            time.sleep(1)
    
    print("[startup] llama.cpp server failed to start, using PyTorch fallback")
    return None


def generate_llama_cpp(prompt: str, max_tokens: int = 300) -> str:
    """Generate response using llama.cpp server (OpenAI-compatible endpoint)."""
    response = requests.post(
        f"{LLAMA_SERVER_URL}/v1/completions",
        json={
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.95,
            "repeat_penalty": 1.1,
            "stop": ["<|user|>", "<|end|>", "\nUser:", "\nHuman:"],
        },
        timeout=60
    )
    return response.json()["choices"][0]["text"].strip()


# Try to use llama.cpp, fall back to PyTorch
llama_proc = start_llama_server()
USE_LLAMA_CPP = llama_proc is not None
```

---

## Expected Performance Comparison

| Backend | Model | Tokens/Second | Memory | Setup |
|---------|-------|---------------|--------|-------|
| PyTorch CPU | GPT-2 base | 8–15 tok/s | ~500MB | Current |
| PyTorch CPU | GPT-2 + LoRA | 6–12 tok/s | ~550MB | After Phase 0 |
| llama.cpp | Orb Q4_K_M | **20–40 tok/s** | **~80MB** | After GGUF |
| llama.cpp | Orb Q8_0 | 15–25 tok/s | ~130MB | After GGUF |

**llama.cpp gives us 2–3× the speed at 15% of the memory.**

---

## Full Pipeline: End-to-End Commands

```bash
# 1. Install dependencies
uv pip install peft datasets transformers torch tqdm gguf sentencepiece

# 2. Download training data
python scripts/download_dataset.py

# 3. Train (run overnight)
python scripts/train_sft_lora.py
python scripts/star_loop.py

# 4. Merge LoRA into base
python scripts/merge_lora.py

# 5. Clone & build llama.cpp
git clone --depth=1 https://github.com/ggerganov/llama.cpp
make -C llama.cpp -j$(nproc)

# 6. Convert to GGUF
python scripts/convert_to_gguf.py
# Produces: models/orb-f16.gguf, models/orb-Q4_K_M.gguf

# 7. Test inference
./llama.cpp/llama-cli -m models/orb-Q4_K_M.gguf \
    -p "You are Orb. User: Tell me about yourself. Orb:" -n 150

# 8. Update app.py to use llama.cpp backend
# (swap the import and model loading section)

# 9. Restart Gradio app
# python app.py
```

---

## Distribution

Once you have `orb-Q4_K_M.gguf`, you can distribute Orb as:

1. **Single file download** — users drop it into any llama.cpp-compatible app
2. **Ollama model** — create a `Modelfile` and push to Ollama Hub:
   ```
   FROM ./models/orb-Q4_K_M.gguf
   SYSTEM "You are Orb, an advanced AI assistant..."
   ```
   ```bash
   ollama create orb -f Modelfile
   ollama run orb
   ```
3. **Hugging Face Hub** — upload the GGUF files with a model card
4. **Self-hosted API** — llama.cpp server with OpenAI-compatible endpoints

**This is how Orb becomes a deployable, portable, open model.**
