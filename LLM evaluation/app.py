# ============================
# app.py — LLM Evaluator
# ============================

import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv

from groq import Groq
import google.generativeai as genai

import nltk
from rouge_score import rouge_scorer

import pandas as pd
import re

# --------------------------------------------------
# Streamlit page config
# --------------------------------------------------
st.set_page_config(
    page_title="LLM Evaluator",
    page_icon="🤖",
    layout="wide"
)

# --------------------------------------------------
# Load .env
# --------------------------------------------------
ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --------------------------------------------------
# Validate API keys
# --------------------------------------------------
if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found. Check your .env file.")
    st.stop()

if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY not found. Check your .env file.")
    st.stop()

# --------------------------------------------------
# Initialize clients
# --------------------------------------------------
groq_client = Groq(api_key=GROQ_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)

# --------------------------------------------------
# Ensure nltk data
# --------------------------------------------------
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

# --------------------------------------------------
# Session state
# --------------------------------------------------
if "responses" not in st.session_state:
    st.session_state.responses = []

if "evaluations" not in st.session_state:
    st.session_state.evaluations = []

# --------------------------------------------------
# App Title
# --------------------------------------------------
st.title("🤖 LLM Evaluator")
st.write("Evaluate QA outputs using Exact Match, F1, and ROUGE")

# --------------------------------------------------
# Evaluation Functions
# --------------------------------------------------
def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return text

def exact_match_score(reference: str, candidate: str):
    return 1 if normalize_text(reference) == normalize_text(candidate) else 0

def f1_token_overlap(reference: str, candidate: str):
    ref_tokens = normalize_text(reference).split()
    cand_tokens = normalize_text(candidate).split()
    if not ref_tokens or not cand_tokens:
        return 0.0
    common = set(ref_tokens) & set(cand_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(cand_tokens)
    recall = len(common) / len(ref_tokens)
    return round((2 * precision * recall) / (precision + recall), 4)

def rouge_scores(reference: str, candidate: str):
    try:
        scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        scores = scorer.score(reference, candidate)
        return {
            "rouge_1": round(scores["rouge1"].fmeasure, 4),
            "rouge_l": round(scores["rougeL"].fmeasure, 4),
        }
    except Exception as e:
        st.warning(f"ROUGE error: {e}")
        return {"rouge_1": 0.0, "rouge_l": 0.0}

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.header("⚙️ Model Configuration")

    model_choice = st.selectbox(
        "Choose a model",
        [
            "Llama 3.3 70B (Groq)",
            "Llama 3.1 8B (Groq)",
            "Qwen 3 32B (Groq)",
            "Gemini 2.5 Flash"
        ]
    )

    st.info("Ready to test models!")

# ==================================================
# TAB 1 — Generate Response
# ==================================================
tab1, tab2, tab3 = st.tabs(["Generate Response", "Evaluate Response", "Analytics"])

with tab1:
    st.subheader("Question Answering")

    context = st.text_area("Context (optional)", height=160)
    question = st.text_input("Question")
    reference_input = st.text_area(
        "Reference Answer (for evaluation only — not sent to model)",
        height=100
    )

    col1, col2 = st.columns(2)
    temperature = col1.slider("Temperature", 0.0, 1.0, 0.7)
    max_tokens = col2.slider("Max Tokens", 100, 1000, 500)

    if st.button("Generate Response", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Generating..."):
                try:
                    if "Groq" in model_choice:
                        model_map = {
                            "Llama 3.3 70B": "llama-3.3-70b-versatile",
                            "Llama 3.1 8B": "llama-3.1-8b-instant",
                            "Qwen 3 32B": "qwen/qwen3-32b",
                        }
                        model_name = next(
                            (v for k, v in model_map.items() if k in model_choice),
                            list(model_map.values())[0],
                        )
                        messages = [
                            {"role": "system", "content": "You are a helpful QA assistant."},
                            {"role": "user", "content": f"Context:\n{context}\n\nQ: {question}"}
                        ]
                        response = groq_client.chat.completions.create(
                            model=model_name,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
                        llm_response = response.choices[0].message.content
                    else:
                        model = genai.GenerativeModel("gemini-2.5-flash")
                        qa_prompt = (
                            f"You are a helpful QA assistant.\nContext:\n{context}\nQuestion:\n{question}"
                        )
                        llm_response = model.generate_content(qa_prompt).text

                    st.success("Generated ✅")
                    st.info(llm_response)

                    st.session_state.responses.append({
                        "model": model_choice,
                        "context": context,
                        "question": question,
                        "reference": reference_input,
                        "response": llm_response,
                    })
                except Exception as e:
                    st.error(f"Error: {e}")

# ==================================================
# TAB 2 — Evaluate Response
# ==================================================
with tab2:
    st.subheader("Evaluate QA Outputs")

    if not st.session_state.responses:
        st.info("Generate a QA answer first in the Generate tab.")
    else:
        idx = st.selectbox(
            "Select saved response",
            range(len(st.session_state.responses)),
            format_func=lambda i: f"{st.session_state.responses[i]['model']} — {st.session_state.responses[i]['question'][:40]}..."
        )
        selected = st.session_state.responses[idx]

        st.write("**Context:**", selected["context"] or "(none)")
        st.write("**Question:**", selected["question"])
        st.write("**Model Answer:**", selected["response"])

        ground_truth = st.text_area(
            "Reference Answer (for evaluation)",
            value=selected.get("reference", "")
        )

        if st.button("Evaluate"):
            if not ground_truth.strip():
                st.warning("Provide reference answer for metrics.")
            else:
                em = exact_match_score(ground_truth, selected["response"])
                f1 = f1_token_overlap(ground_truth, selected["response"])
                rouge = rouge_scores(ground_truth, selected["response"])

                col1, col2, col3 = st.columns(3)
                col1.metric("Exact Match", em)
                col2.metric("F1 Token Overlap", f1)
                col3.metric("ROUGE-L", rouge["rouge_l"])

                st.session_state.evaluations.append({
                    "model": selected["model"],
                    "question": selected["question"],
                    "exact_match": em,
                    "f1": f1,
                    "rouge_l": rouge["rouge_l"],
                })

                st.success("Saved Evaluation")

# ==================================================
# TAB 3 — Analytics
# ==================================================
with tab3:
    if not st.session_state.evaluations:
        st.info("No evaluations yet.")
    else:
        df = pd.DataFrame(st.session_state.evaluations)
        st.subheader("Evaluation Summary")
        st.dataframe(df, use_container_width=True)

        st.subheader("Model Comparison")
        st.bar_chart(df.groupby("model")[["exact_match","f1","rouge_l"]].mean())
