

# Multilingual Legal Document Analyzer



This project is a Streamlit-based multilingual legal assistant that allows users to upload legal documents in various languages, extract and summarize their contents, detect and display named entities, and engage in an intelligent Q\&A chat about the document or general legal topics. It uses Llama models via LM Studio for natural language interaction.

---

## Features

* **Document Upload (PDF, DOCX, TXT)**
* **Automatic Language Detection**
  All outputs and UI dynamically adapt to the language of the document.
* **Named Entity Recognition**
  Key legal entities are extracted and highlighted using spaCy.
* **Legal Document Summarization**
  Summarized using multilingual Transformer models (T5 / mBART).
* **Multilingual Chat Interface**
  Interact with a LLaMA-based assistant (via LM Studio) to ask questions about the document or general legal topics.
* **Smart UI with Chat History**
  Clean Streamlit interface with saved chats and back-and-forth chat bubbles.

---



## Tech Stack

| Component          | Tool/Library                                 |
| ------------------ | -------------------------------------------- |
| UI                 | Streamlit                                    |
| Language Detection | `langdetect`                                 |
| Entity Extraction  | spaCy (`en_core_web_sm`, `xx_ent_wiki_sm`)   |
| Summarization      | Hugging Face Transformers (e.g., T5, mBART)  |
| Chat Model         | LLaMA (via [LM Studio](https://lmstudio.ai)) |
| PDF/Text Parsing   | PyMuPDF, python-docx                         |
| Backend Logic      | Python                                       |

---

## Installation

```bash
git clone https://github.com/Habiba480/legal-document-analyzer.git
cd legal-document-analyzer
pip install -r requirements.txt
```

Then download required spaCy models:

```bash
python -m spacy download en_core_web_sm
python -m spacy download xx_ent_wiki_sm
```

---

## Usage

1. Launch LM Studio and load your preferred multilingual LLaMA model.
2. Start the Streamlit app:

```bash
streamlit run app.py
```

3. Upload a legal document (PDF, TXT, DOCX).
4. View extracted text (toggle dropdown), named entities, and document summary.
5. Ask questions about the document or general legal issues in the detected language.

---

## File Structure

```
legal-document-analyzer/
│
├── app.py                      # Main Streamlit app
├── requirements.txt           # Dependencies
└── README.md
```

---

## Notes

* The app currently supports multiple languages, but language-dependent tasks (e.g., NER) vary in performance based on spaCy model coverage.
* Chat completion is powered by LM Studio; ensure it is running before chatting.

---

## License

This project is open source and free to use under the [MIT License](LICENSE).


