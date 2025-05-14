import streamlit as st
from transformers import pipeline
import pandas as pd
import random
import json

# --- Page & model setup ---
st.set_page_config(page_title="Creepy GPT", layout="centered")
st.title("👻 Creepy GPT")
st.write("Branching horror story generator")

@st.cache_resource
def load_generator():
    return pipeline("text-generation", model="distilgpt2")
gen = load_generator()

# --- Load seeds.json ---
@st.cache_data
def load_seeds(path="seeds.json"):
    data = json.load(open(path, "r", encoding="utf-8"))
    df = pd.DataFrame(data)
    if "intensity" not in df.columns:
        df["intensity"] = 3
    return df

df = load_seeds()

# --- Initialize story history ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- Display story so far ---
for i, part in enumerate(st.session_state.history):
    st.markdown(f"**Part {i+1}:** {part}")

# --- UI: slider, dropdown ---
lvl = st.slider("Max horror level (1=mild … 5=extreme)", 1, 5, 3)
st.markdown(f"Showing random seeds up to **level {lvl}**")

cands = df[df["intensity"] <= lvl][["text","intensity"]].to_dict(orient="records")
random.shuffle(cands)
cands = cands[:10]

options = [(f"{c['text']}  — level {c['intensity']}", c["text"]) for c in cands]
labels, values = zip(*options)
idx = st.selectbox("Pick your scary seed", range(len(labels)),
                   format_func=lambda i: labels[i])
seed = values[idx]

# --- Single “Spook me” handler ---
if st.button("Spook me"):
    # Build a prompt that includes all previous parts + the new seed
    context = "\n\n".join(st.session_state.history)
    prompt = (f"{context}\n\nContinue the horror story. Next: {seed}"
              if context else seed)
    with st.spinner("Building the nightmare…"):
        out = gen(prompt, max_length=150, do_sample=True, top_p=0.9)[0]["generated_text"]
    # Save & display
    st.session_state.history.append(out)
    st.write(out)

    # Scare‐score on the most recent part
    horror_lex = ["blood","death","ghost","corpse","dark","scream","shadow","fear"]
    score = min(100, sum(out.lower().count(w) for w in horror_lex) * 10)
    st.markdown(f"**Scare score:** {score}/100")
