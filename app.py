import os
import streamlit as st
import fitz  # PyMuPDF
import docx2txt
import torch
from dotenv import load_dotenv
from transformers import (
    pipeline,
    T5Tokenizer,
    T5ForConditionalGeneration,
    MarianMTModel,
    MarianTokenizer,
    AutoModelForTokenClassification,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# MODEL LOADERS
@st.cache_resource
def load_english_models():
    ner = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    summarizer = T5ForConditionalGeneration.from_pretrained("t5-base")
    return ner, tokenizer, summarizer

@st.cache_resource
def load_arabic_models():
    ner_model = "CAMeL-Lab/bert-base-arabic-camelbert-msa-ner"
    ner_tok = AutoTokenizer.from_pretrained(ner_model, use_auth_token=HF_TOKEN)
    ner_mod = AutoModelForTokenClassification.from_pretrained(ner_model, use_auth_token=HF_TOKEN)
    ner = pipeline("ner", model=ner_mod, tokenizer=ner_tok, grouped_entities=True)

    summ_model = "csebuetnlp/mT5_multilingual_XLSum"
    summ_tok = AutoTokenizer.from_pretrained(summ_model)
    summ_mod = AutoModelForSeq2SeqLM.from_pretrained(summ_model)

    return ner, summ_tok, summ_mod

@st.cache_resource
def load_translation_models():
    ar_to_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
    ar_to_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
    en_to_ar_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
    en_to_ar_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
    return ar_to_en_tokenizer, ar_to_en_model, en_to_ar_tokenizer, en_to_ar_model

@st.cache_resource
def load_language_detector():
    return pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")

@st.cache_resource
def load_multilingual_classifier():
    model_name = "joeddav/xlm-roberta-large-xnli"
    classifier = pipeline("zero-shot-classification", model=model_name)
    return classifier

#  UTILITIES
def extract_text(file):
    if file.name.endswith(".pdf"):
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return "\n".join([page.get_text() for page in doc])
    elif file.name.endswith(".docx"):
        return docx2txt.process(file)
    return ""

def extract_entities(text, ner_pipeline, filter_types=None):
    raw = ner_pipeline(text[:1000])
    entities = [(ent["word"], ent["entity_group"]) for ent in raw]
    if filter_types:
        entities = [e for e in entities if e[1] in filter_types]
    return entities

def summarize(text, tokenizer, model, is_arabic=False):
    prefix = "" if is_arabic else "summarize: "
    inputs = tokenizer.encode(prefix + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=40, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def translate(text, tokenizer, model):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**tokens)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

#UI
st.set_page_config(page_title="Legal Document Analyzer", layout="wide")

st.markdown("""
    <div style='text-align: center; padding: 1rem 0'>
        <h1 style='font-size: 2.5rem;'>Legal Document Analyzer</h1>
        <p style='font-size: 1.2rem; color: gray;'>Summarize, classify, extract entities, and translate legal documents in English and Arabic.</p>
    </div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a legal document (PDF or DOCX)", type=["pdf", "docx"])

if uploaded_file:
    text = extract_text(uploaded_file)

    detector = load_language_detector()
    lang_label = detector(text[:512])[0]['label']
    language = "Arabic" if lang_label == "ar" else "English"

    st.markdown("### Extracted Text")
    with st.expander("Click to expand extracted text", expanded=False):
        st.write(text[:3000] + ("..." if len(text) > 3000 else ""))

    if language == "English":
        ner, tokenizer, summarizer_model = load_english_models()
        entity_types = ['PERS', 'ORG', 'MISC', 'LOC']
        is_arabic = False
    else:
        ner, tokenizer, summarizer_model = load_arabic_models()
        entity_types = ['PERS', 'ORG', 'MISC', 'LOC']
        is_arabic = True

    classifier = load_multilingual_classifier()

    st.markdown("---")

    st.markdown("### Document Type Classification")

    if language == "Arabic":
        candidate_labels = [
            "عقد", "اتفاقية", "أمر محكمة", "ترخيص", "سياسة",
            "مذكرة", "إشعار", "خطاب قانوني", "تنظيم", "وثيقة براءة اختراع"
        ]
    else:
        candidate_labels = [
            "Contract", "Agreement", "Court Order", "License", "Policy",
            "Memorandum", "Notice", "Legal Letter", "Regulation", "Patent Document"
        ]

    result = classifier(text[:1000], candidate_labels=candidate_labels)
    predicted_label = result['labels'][0]
    confidence = result['scores'][0]

    st.success(f"Predicted Type: **{predicted_label}** ({confidence*100:.1f}% confidence)")

    st.markdown("#### Top Predictions:")
    for lbl, score in zip(result['labels'][:3], result['scores'][:3]):
        st.markdown(f"- **{lbl}**: {score*100:.2f}%")

    st.markdown("---")

    st.markdown("### Named Entity Recognition (NER)")
    entity_filter = st.multiselect("Filter by entity type", options=entity_types)
    entities = extract_entities(text, ner, filter_types=entity_filter if entity_filter else None)

    if entities:
        st.markdown("Detected Entities:")
        for ent_text, ent_label in entities:
            st.markdown(f"- **{ent_text}** *(type: {ent_label})*")

        found_entity_types = set(ent_label for _, ent_label in entities)

        entity_type_explanations_en = {
            'PERS': "Person — A named individual such as a client, lawyer, judge, or witness.",
            'ORG': "Organization — A company, agency, court, or other formal group.",
            'LOC': "Location — A named place such as a country, city, state, or region.",
            'MISC': "Miscellaneous — Other named entities like events, nationalities, laws."
        }
        entity_type_explanations_ar = {
            'PERS': "شخص — اسم فرد مثل عميل أو محامٍ أو قاضٍ أو شاهد.",
            'ORG': "منظمة — مؤسسة أو شركة أو وكالة أو محكمة أو مجموعة رسمية.",
            'LOC': "موقع — مكان جغرافي مثل دولة أو مدينة أو ولاية أو منطقة.",
            'MISC': "كيانات متنوعة — مثل الأحداث أو الجنسيات أو القوانين."
        }

        st.markdown("### Explanation of Entity Types")
        if language == "Arabic":
            for etype in found_entity_types:
                explanation = entity_type_explanations_ar.get(etype, "لا يوجد وصف.")
                st.markdown(f"- **{etype}**: {explanation}")
        else:
            for etype in found_entity_types:
                explanation = entity_type_explanations_en.get(etype, "No description available.")
                st.markdown(f"- **{etype}**: {explanation}")

    else:
        st.info("No entities found for the selected filters.")

    st.markdown("---")

    st.markdown("### Document Summary")
    if st.button("Generate Summary"):
        summary = summarize(text, tokenizer, summarizer_model, is_arabic=is_arabic)
        st.success("Summary Generated:")
        st.text_area("Summary", summary, height=200)
        st.download_button("Download Summary", summary, file_name="summary.txt")

    st.markdown("---")

    st.markdown("### Document Translation")
    ar_to_en_tokenizer, ar_to_en_model, en_to_ar_tokenizer, en_to_ar_model = load_translation_models()

    if language == "Arabic":
        if st.button("Translate to English"):
            translated = translate(text[:1000], ar_to_en_tokenizer, ar_to_en_model)
            st.text_area("English Translation", translated, height=200)
    else:
        if st.button("Translate to Arabic"):
            translated = translate(text[:1000], en_to_ar_tokenizer, en_to_ar_model)
            st.text_area("Arabic Translation", translated, height=200)

st.markdown("<hr><center>Made with ❤️by Habiba</center>", unsafe_allow_html=True)
