import streamlit as st
from transformers import pipeline
import pandas as pd
import random
import json

st.set_page_config(page_title="Creepy GPT", layout="centered")
st.title("👻 Creepy GPT")
st.write("Branching horror story generator")

@st.cache_resource
def load_generator():
    return pipeline("text-generation", model="distilgpt2")
gen = load_generator()

@st.cache_data
def load_seeds(path="seeds.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# Load & prepare seeds
seeds = load_seeds()  
df = pd.DataFrame(seeds)
if "intensity" not in df.columns:
    df["intensity"] = 3  # default intensity

# Slider & filter
lvl = st.slider("Max horror level (1=mild … 5=extreme)", 1, 5, 3)
st.markdown(f"Showing random seeds up to **level {lvl}**")

filtered = df[df["intensity"] <= lvl][["text","intensity"]].to_dict(orient="records")
random.shuffle(filtered)
choices = filtered[:10]

options = [(f"{c['text']}  — level {c['intensity']}", c["text"]) for c in choices]
labels, values = zip(*options)
idx = st.selectbox("Pick your scary seed", list(range(len(labels))),
                   format_func=lambda i: labels[i])
seed = values[idx]

if st.button("Spook me"):
    out = gen(seed, max_length=150, do_sample=True, top_p=0.9)[0]["generated_text"]
    st.write(out)
    lex = ["blood","death","ghost","corpse","dark","scream","shadow","fear"]
    score = min(100, sum(out.lower().count(w) for w in lex) * 10)
    st.markdown(f"**Scare score:** {score}/100")
