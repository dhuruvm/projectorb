"""Download DialoGPT-medium to models/dialogpt-medium for Orb base."""
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

OUT = "models/dialogpt-medium"
os.makedirs(OUT, exist_ok=True)
print("Downloading DialoGPT-medium (conversational GPT-2, ~1.4 GB)…")
tok = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
mdl = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
tok.save_pretrained(OUT)
mdl.save_pretrained(OUT)
print(f"Done → {OUT}")
