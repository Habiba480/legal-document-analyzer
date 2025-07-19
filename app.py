import os
import json
import requests
import streamlit as st
import fitz
import docx2txt
import spacy
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from langdetect import detect
from langcodes import Language

st.set_page_config(page_title="Legal Document Analyzer", layout="wide")
if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = []

if "current_chat" not in st.session_state:
    st.session_state.current_chat = []

if "selected_chat" not in st.session_state:
    st.session_state.selected_chat = None

if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = []
if "current_chat" not in st.session_state:
    st.session_state.current_chat = []
if "current_doc_text" not in st.session_state:
    st.session_state.current_doc_text = ""
if "current_language" not in st.session_state:
    st.session_state.current_language = "en"

# Load models
@st.cache_resource
def load_models():
    summarizer = T5ForConditionalGeneration.from_pretrained("t5-base")
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    spacy_model = spacy.load("en_core_web_sm")
    return summarizer, tokenizer, spacy_model

summarizer, tokenizer, nlp = load_models()

st.title("Legal Document Analyzer")

uploaded_file = st.file_uploader("Upload a legal document (PDF/DOCX)", type=["pdf", "docx"])

if uploaded_file:
    if uploaded_file.type == "application/pdf":
        # Use PyMuPDF (fitz) correctly to extract text
        pdf_doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = ""
        for page in pdf_doc:
            text += page.get_text()
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        # Use docx2txt to extract text from DOCX
        import docx2txt
        text = docx2txt.process(uploaded_file)
    else:
        st.error("Unsupported file format")
        st.stop()

    language = detect(text)
    lang_name = Language.get(language).display_name()
    st.session_state.current_language = language
    st.session_state.current_doc_text = text

    st.markdown(f"**Detected Language:** {lang_name}")

    with st.expander("📄 Show Extracted Text"):
        st.write(text)

    # Summarization
    if st.button("Summarize Document"):
        inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = summarizer.generate(inputs, max_length=150, min_length=40, length_penalty=5., num_beams=2)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        st.subheader("Summary")
        st.write(summary)

    # Entity extraction
    doc = nlp(text)
    if doc.ents:
        st.subheader("Extracted Entities")
        for ent in doc.ents:
            st.markdown(f"**{ent.label_}:** {ent.text}")
    else:
        st.info("No entities detected.")

    # Chat interface
    st.subheader("Ask Questions About the Document")
    user_input = st.chat_input("Ask a legal question")
    if user_input:
        st.session_state.current_chat.append(("user", user_input))
        with st.spinner("Thinking..."):
            prompt = f"You are a legal assistant. The user uploaded this document:\n\n{text}\n\nQuestion: {user_input}\n\nAnswer in {lang_name}:"
            res = requests.post("http://localhost:1234/v1/chat/completions", headers={"Content-Type": "application/json"}, data=json.dumps({
                "model": "llama",
                "messages": [
                    {"role": "system", "content": "You are a helpful legal assistant."},
                    {"role": "user", "content": prompt}
                ]
            }))
            reply = res.json()['choices'][0]['message']['content']
            st.session_state.current_chat.append(("assistant", reply))

    for role, msg in st.session_state.current_chat:
        with st.chat_message(role):
            st.markdown(msg)

with st.sidebar:
    st.header("Chats")

    if st.session_state.chat_histories:
        chat_names = [chat["title"] for chat in st.session_state.chat_histories]

        # Make sure there's a valid default selection
        if "selected_chat" not in st.session_state:
            st.session_state.selected_chat = chat_names[-1]

        selected_chat = st.radio("Select chat", options=chat_names, index=chat_names.index(st.session_state.selected_chat))

        if st.button("New Chat"):
            # Save the current chat before starting a new one
            if st.session_state.current_chat:
                st.session_state.chat_histories.append({
                    "title": st.session_state.current_chat[0]["content"][:30] + "..." if st.session_state.current_chat else "Untitled",
                    "messages": st.session_state.current_chat
                })

            st.session_state.current_chat = []
            st.session_state.selected_chat = None  # Reset selection
            st.rerun()
    else:
        if st.button("New Chat"):
            st.session_state.current_chat = []
            st.session_state.selected_chat = None
            st.rerun()
