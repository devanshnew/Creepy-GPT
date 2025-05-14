import streamlit as st
from transformers import pipeline
import pandas as pd
import random
import json

# --- Page setup ---
st.set_page_config(page_title="Creepy GPT", layout="centered")
st.title("👻 Creepy GPT")
st.write("Branching horror story generator")

# --- Load the model once ---
@st.cache_resource
def load_generator():
    return pipeline("text-generation", model="distilgpt2")

gen = load_generator()

# --- Load all seeds from your local JSON file ---
@st.cache_data
def load_seeds(path="seeds.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)

df = load_seeds()  # expects [{'text':..., 'intensity':1–5}, ...]

# --- Horror‐level slider ---
intensity = st.slider(
    "Max horror level", 
    min_value=1, max_value=5, value=3,
    help="1=mild … 5=extreme"
)
st.markdown(f"Showing random seeds up to **level {intensity}**")

# --- Filter & random subset of 10 seeds ---
filtered = df[df["intensity"] <= intensity][["text", "intensity"]].to_dict(orient="records")
random.shuffle(filtered)
choices = filtered[:10]

# --- Dropdown with annotated levels ---
options = [
    (f"{item['text']}  — level {item['intensity']}", item["text"])
    for item in choices
]
display, raw = zip(*options)
idx = st.selectbox("Pick your scary seed", options=list(range(len(display))),
                   format_func=lambda i: display[i])
seed = raw[idx]

# --- Generate & display snippet + scare‐score ---
if st.button("Spook me"):
    with st.spinner("Summoning the horror..."):
        result = gen(seed, max_length=150, do_sample=True, top_p=0.9)[0]["generated_text"]
    st.write(result)

    # simple lexicon‐based scare score
    horror_lexicon = ["blood","death","ghost","corpse","dark","scream","shadow","fear"]
    score = sum(result.lower().count(w) for w in horror_lexicon)
    score = min(100, score * 10)
    st.markdown(f"**Scare score:** {score}/100")
