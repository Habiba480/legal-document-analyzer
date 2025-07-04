# Legal Document Analyzer

**Legal Document Analyzer** is an interactive web application built with Streamlit that helps users process and analyze legal documents in English and Arabic. It supports language detection, named entity recognition (NER), document classification, summarization, and translation between Arabic and English.

---

## Features

- **Language Detection**
  Automatically identifies whether the uploaded legal document is written in English or Arabic.

- **Named Entity Recognition (NER)**
  Detects and classifies named entities such as people, organizations, locations, and miscellaneous items using language-specific transformer models.

- **Text Summarization**
  Generates concise summaries of lengthy legal documents using T5 and mT5-based models.

- **Document Classification**
  Classifies the type of legal document (e.g., contract, agreement, policy) using zero-shot classification.

- **Translation**
  Enables translation of legal text between English and Arabic using MarianMT models.

- **File Format Support**
  Accepts both `.pdf` and `.docx` file uploads.

---

## Technology Stack

- **Frontend:** Streamlit
- **NLP Models (via Hugging Face Transformers):**
  - `dslim/bert-base-NER` (English NER)
  - `CAMeL-Lab/bert-base-arabic-camelbert-msa-ner` (Arabic NER)
  - `t5-base` (English summarization)
  - `csebuetnlp/mT5_multilingual_XLSum` (Arabic summarization)
  - `Helsinki-NLP/opus-mt-ar-en`, `opus-mt-en-ar` (Translation)
  - `joeddav/xlm-roberta-large-xnli` (Zero-shot classification)
  - `papluca/xlm-roberta-base-language-detection` (Language detection)
- **Others:** PyMuPDF, docx2txt, dotenv

---

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Habiba480/legal-document-analyzer.git
   cd legal-document-analyzer
