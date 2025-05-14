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
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    if "intensity" not in df.columns:
        df["intensity"] = 3
    return df

df = load_seeds()

# --- UI: slider & dropdown ---
lvl = st.slider("Max horror level (1=mild … 5=extreme)", 1, 5, 3)
st.markdown(f"Showing random seeds up to **level {lvl}**")

choices = df[df["intensity"] <= lvl][["text","intensity"]].to_dict(orient="records")
random.shuffle(choices)
choices = choices[:10]

options = [(f"{c['text']}  — level {c['intensity']}", c["text"]) for c in choices]
labels, values = zip(*options)
sel = st.selectbox("Pick your scary seed", list(range(len(labels))),
                   format_func=lambda i: labels[i])
seed = values[sel]

# --- Handle button press & store in session ---
if "story" not in st.session_state:
    st.session_state.story = []  # holds generated snippets

if st.button("Spook me"):
    st.session_state.last_action = "button_pressed"
    try:
        with st.spinner("Summoning terror…"):
            out = gen(seed, max_length=150, do_sample=True, top_p=0.9)[0]["generated_text"]
        # save
        st.session_state.story.append(out)
        st.session_state.last_error = None
    except Exception as e:
        st.session_state.last_error = str(e)

# --- Debug & display ---
if st.session_state.get("last_action") == "button_pressed":
    st.markdown("**[Debug]** Button registered")
if st.session_state.get("last_error"):
    st.error(f"Generation error: {st.session_state.last_error}")

# --- Show the story so far ---
for i, snippet in enumerate(st.session_state.story):
    st.markdown(f"**Part {i+1}:** {snippet}")

# --- Compute & show scare score for the last snippet ---
if st.session_state.story:
    last = st.session_state.story[-1]
    horror_lex = ["blood","death","ghost","corpse","dark","scream","shadow","fear"]
    score = min(100, sum(last.lower().count(w) for w in horror_lex)*10)
    st.markdown(f"**Scare score:** {score}/100")
