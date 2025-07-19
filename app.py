import os
import fitz
import docx2txt
import streamlit as st
from langdetect import detect
import spacy
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("SpaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
    st.stop()

llm = ChatOpenAI(
    base_url="http://0.0.0.0:1234/v1",
    model="meta-llama-3.1-8b-instruct",
    api_key="lm-studio"
)

st.set_page_config(page_title="Legal Document Chatbot", layout="wide")

UI_STRINGS = {
    "en": {
        "title": "Legal Document Analyzer & Chatbot",
        "upload_label": "Upload legal document (PDF or DOCX)",
        "show_text": "Show extracted text",
        "detected_lang": "Detected Language",
        "entities_header": "Extracted Entities",
        "no_entities": "No entities detected.",
        "summary_header": "Document Summary",
        "ask_header": "Ask Questions About The Document",
        "question_input": "Enter your question about this document or legal topics:",
        "new_chat": "➕ New Chat",
        "select_chat": "Select chat",
        "load_chat": "🔁 Load Selected Chat",
        "unsupported_file": "Unsupported file format",
        "typing": "🤖 is typing...",
        "thinking": "Fetching answer...",
    },
    "ar": {
        "title": "محلل ومستشار المستندات القانونية",
        "upload_label": "قم بتحميل مستند قانوني (PDF أو DOCX)",
        "show_text": "عرض النص المستخرج",
        "detected_lang": "اللغة المكتشفة",
        "entities_header": "الكيانات المستخرجة",
        "no_entities": "لم يتم اكتشاف كيانات.",
        "summary_header": "ملخص المستند",
        "ask_header": "اسأل عن المستند",
        "question_input": "أدخل سؤالك عن هذا المستند أو المواضيع القانونية:",
        "new_chat": "➕ محادثة جديدة",
        "select_chat": "اختر المحادثة",
        "load_chat": "🔁 تحميل المحادثة المحددة",
        "unsupported_file": "نوع الملف غير مدعوم",
        "typing": "🤖 جارِ الكتابة...",
        "thinking": "جاري جلب الإجابة...",
    }
}

def get_ui_strings(lang_code):
    if lang_code not in UI_STRINGS:
        return UI_STRINGS["en"]
    return UI_STRINGS[lang_code]

# Session State Init
if "current_chat" not in st.session_state:
    st.session_state.current_chat = []

if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = []

if "chat_titles" not in st.session_state:
    st.session_state.chat_titles = []

# Helper to generate chat title from first user message
def generate_chat_title(chat):
    for msg in chat:
        if msg["role"] == "user" and msg["content"].strip():
            return msg["content"][:30] + "..." if len(msg["content"]) > 30 else msg["content"]
    return "Untitled Chat"

# Upload file and extract text
uploaded_file = st.file_uploader("Upload legal document (PDF or DOCX)", type=["pdf", "docx"])

full_text = None
detected_lang_code = "en"
ui = UI_STRINGS["en"]  # default UI strings

if uploaded_file:
    if uploaded_file.type == "application/pdf":
        pdf_doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        full_text = ""
        for page in pdf_doc:
            full_text += page.get_text()
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        full_text = docx2txt.process(uploaded_file)
    else:
        st.error("Unsupported file format")
        st.stop()

    detected_lang_code = detect(full_text)[:2]
    ui = get_ui_strings(detected_lang_code)

    st.title(ui["title"])

    with st.expander(ui["show_text"]):
        st.text_area(ui["show_text"], full_text, height=300)

    st.markdown(f"**{ui['detected_lang']}:** {detected_lang_code}")

    doc = nlp(full_text)
    ents = [(ent.text, ent.label_) for ent in doc.ents]

    st.markdown(f"### {ui['entities_header']}")
    if ents:
        for ent_text, ent_label in ents:
            st.write(f"- {ent_text} ({ent_label})")
    else:
        st.write(ui["no_entities"])

    summary_prompt = f"Summarize this legal document briefly:\n\n{full_text[:2000]}"
    with st.spinner(ui["thinking"]):
        summary = llm.predict(summary_prompt)
    st.markdown(f"### {ui['summary_header']}")
    st.write(summary)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(full_text)

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_texts(chunks, embedding_model, persist_directory="./legal_doc_db")
    vectorstore.persist()

    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )

    # Sidebar with chat history and new chat button
    with st.sidebar:
        st.header("Chats")

        if st.button(ui["new_chat"]):
            if st.session_state.current_chat:
                title = generate_chat_title(st.session_state.current_chat)
                st.session_state.chat_sessions.append(st.session_state.current_chat.copy())
                st.session_state.chat_titles.append(title)
                st.session_state.current_chat = []
                st.experimental_rerun()

        if st.session_state.chat_sessions:
            selected_title = st.radio(ui["select_chat"], st.session_state.chat_titles, index=len(st.session_state.chat_titles) - 1)
            if selected_title in st.session_state.chat_titles:
                selected_index = st.session_state.chat_titles.index(selected_title)

                if st.button(ui["load_chat"]):
                    st.session_state.current_chat = st.session_state.chat_sessions[selected_index].copy()
                    st.experimental_rerun()

    # Main chat area: render messages and input
    for msg in st.session_state.current_chat:
        icon = "👤" if msg["role"] == "user" else "🤖"
        css_class = "user-msg" if msg["role"] == "user" else "bot-msg"
        st.markdown(f'<div class="{css_class}">{icon} {msg["content"]}</div>', unsafe_allow_html=True)

    query = st.chat_input(ui["question_input"])

    if query:
        st.session_state.current_chat.append({"role": "user", "content": query})

        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(f'<div class="user-msg">{query}</div>', unsafe_allow_html=True)

        # Typing bubble
        with st.chat_message("assistant"):
            typing_placeholder = st.empty()
            typing_placeholder.markdown(f'<div class="typing-bubble">{ui["typing"]}</div>', unsafe_allow_html=True)

        # Get answer from QA chain
        result = qa_chain({"query": query})
        answer = result["result"]

        # Replace typing bubble with answer
        typing_placeholder.markdown(f'<div class="bot-msg">{answer}</div>', unsafe_allow_html=True)

        st.session_state.current_chat.append({"role": "assistant", "content": answer})

else:
    st.title(ui["title"])

# Dark theme CSS
st.markdown("""
<style>
    body, .stApp {
        background-color: #0d0d0d;
        color: #f5f5f5;
    }
    .user-msg {
        background-color: #1a1a1a;
        color: #eaeaea;
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 8px;
        max-width: 70%;
        align-self: flex-end;
    }
    .bot-msg {
        background-color: #262626;
        color: #f0f0f0;
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 8px;
        max-width: 70%;
        align-self: flex-start;
    }
    .typing-bubble {
        background-color: #1f1f1f;
        color: #aaaaaa;
        font-style: italic;
        border-radius: 10px;
        padding: 8px 12px;
        margin-bottom: 10px;
        max-width: fit-content;
        align-self: flex-start;
    }
</style>
""", unsafe_allow_html=True)
