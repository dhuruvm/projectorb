"""Download Llama-3.2-1B-Instruct to models/llama-3.2-1b."""
from transformers import AutoModelForCausalLM, AutoTokenizer
import os, sys

OUT = "models/llama-3.2-1b"
os.makedirs(OUT, exist_ok=True)

# unsloth mirror — same weights as meta-llama/Llama-3.2-1B-Instruct, no HF auth needed
HF_ID = "unsloth/Llama-3.2-1B-Instruct"

print(f"Downloading {HF_ID} → {OUT} …")
sys.stdout.flush()

print("  Tokenizer…")
tok = AutoTokenizer.from_pretrained(HF_ID)
tok.save_pretrained(OUT)
print("  Model weights (~2.4 GB)…")
sys.stdout.flush()

mdl = AutoModelForCausalLM.from_pretrained(HF_ID, torch_dtype="auto")
mdl.save_pretrained(OUT)
print(f"Done. Files: {os.listdir(OUT)}")
