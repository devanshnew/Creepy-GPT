import streamlit as st
from transformers import pipeline
from datasets import load_dataset
import pandas as pd
import random

# --- Title ---
st.title("Creepy GPT")
st.write("Branching horror story generator")

# --- Load the model once ---
gen = pipeline("text-generation", model="distilgpt2")

# --- 1) Fetch the Two-Sentence Horror dataset ---
ds = load_dataset("voacado/bart-two-sentence-horror", split="train")

# --- 2) Convert to DataFrame ---
df = pd.DataFrame(ds)
# The dataset has 'text' and 'upvotes' columns; rename upvotes → score
df = df.rename(columns={"upvotes": "score"})

# --- 3) Quantile-bin the 'score' into 1–5 intensity levels ---
df["intensity"] = pd.qcut(df["score"].fillna(0), 5, labels=False) + 1

# --- 4) Slider for max horror level ---
intensity = st.slider("Max horror level (1=mild … 5=extreme)", 1, 5, 3)
st.write(f"Showing random seeds up to level **{intensity}**")

# --- 5) Filter & randomize a small subset for this session ---
filtered = df[df["intensity"] <= intensity][["text","intensity"]].to_dict(orient="records")
random.shuffle(filtered)
choices = filtered[:10]

# --- 6) Build annotated dropdown options ---
options = [
    (f"{item['text']}  —  level {item['intensity']}", item["text"])
    for item in choices
]
display, raw = zip(*options)

idx = st.selectbox("Pick your scary seed", list(range(len(display))),
                   format_func=lambda i: display[i])
seed = raw[idx]

# --- 7) Generate and display snippet + score ---
if st.button("Spook me"):
    out = gen(seed, max_length=150, do_sample=True, top_p=0.9)[0]["generated_text"]
    st.write(out)

    # Quick lexicon-based “scare score” (optional)
    horror_lexicon = ["blood","death","ghost","corpse","dark","scream","shadow","fear"]
    score = sum(out.lower().count(w) for w in horror_lexicon)
    st.markdown(f"**Scare score:** {min(100, score*10)}/100")
