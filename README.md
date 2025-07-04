# Legal Document Analyzer

A comprehensive multilingual legal document processing application that provides automated analysis, classification, summarization, and translation capabilities for legal documents in English and Arabic.

## Features

###  **Document Analysis**
- **Named Entity Recognition (NER)**: Automatically identifies and extracts legal entities (persons, organizations, locations, miscellaneous entities)
- **Document Classification**: Categorizes documents into legal types (contracts, agreements, court orders, licenses, etc.)
- **Text Extraction**: Supports PDF and DOCX file formats

###  **Multilingual Support**
- **Language Detection**: Automatic detection of document language (English/Arabic)
- **Bilingual NER**: Specialized models for both English and Arabic entity recognition
- **Translation**: Bidirectional translation between English and Arabic

###  **Document Processing**
- **Summarization**: Generates concise summaries using T5 and mT5 models
- **Entity Filtering**: Filter entities by type for focused analysis
- **Export Functionality**: Download summaries and translations

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd legal-document-analyzer
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
Create a `.env` file in the project root:
```env
HF_TOKEN=your_huggingface_token_here
```

4. **Run the application**
```bash
streamlit run app.py
```

## Dependencies

```txt
streamlit
PyMuPDF
docx2txt
torch
python-dotenv
transformers
```

## Models Used

### English Models
- **NER**: `dslim/bert-base-NER` - Named Entity Recognition
- **Summarization**: `t5-base` - Text summarization
- **Translation**: `Helsinki-NLP/opus-mt-en-ar` - English to Arabic translation

### Arabic Models
- **NER**: `CAMeL-Lab/bert-base-arabic-camelbert-msa-ner` - Arabic Named Entity Recognition
- **Summarization**: `csebuetnlp/mT5_multilingual_XLSum` - Multilingual summarization
- **Translation**: `Helsinki-NLP/opus-mt-ar-en` - Arabic to English translation

### Multilingual Models
- **Language Detection**: `papluca/xlm-roberta-base-language-detection`
- **Classification**: `joeddav/xlm-roberta-large-xnli` - Zero-shot classification

## Usage

### 1. Document Upload
- Upload your legal document (PDF or DOCX format)
- The application automatically detects the document language

### 2. Document Classification
- View the predicted document type with confidence scores
- See top predictions for document categorization

### 3. Named Entity Recognition
- Extract entities from the document
- Filter by entity types (Person, Organization, Location, Miscellaneous)
- View explanations for each entity type

### 4. Document Summarization
- Generate automated summaries
- Download summaries as text files

### 5. Translation
- Translate documents between English and Arabic
- View translations in the web interface

## Supported Document Types

### English
- Contract
- Agreement
- Court Order
- License
- Policy
- Memorandum
- Notice
- Legal Letter
- Regulation
- Patent Document

### Arabic
- عقد (Contract)
- اتفاقية (Agreement)
- أمر محكمة (Court Order)
- ترخيص (License)
- سياسة (Policy)
- مذكرة (Memorandum)
- إشعار (Notice)
- خطاب قانوني (Legal Letter)
- تنظيم (Regulation)
- وثيقة براءة اختراع (Patent Document)

## Entity Types

### English
- **PERS**: Person (clients, lawyers, judges, witnesses)
- **ORG**: Organization (companies, agencies, courts)
- **LOC**: Location (countries, cities, states, regions)
- **MISC**: Miscellaneous (events, nationalities, laws)

### Arabic
- **PERS**: شخص (الأشخاص والمحامين والقضاة والشهود)
- **ORG**: منظمة (الشركات والوكالات والمحاكم)
- **LOC**: موقع (البلدان والمدن والولايات والمناطق)
- **MISC**: كيانات متنوعة (الأحداث والجنسيات والقوانين)

## Technical Architecture

### Caching Strategy
- Uses Streamlit's `@st.cache_resource` for efficient model loading
- Models are loaded once and reused across sessions

### Performance Optimization
- Text truncation for processing efficiency
- Batch processing for translations
- Lazy loading of language-specific models

## Configuration

### Environment Variables
- `HF_TOKEN`: Hugging Face API token for accessing gated models

### Model Parameters
- Maximum input length: 512 tokens (language detection)
- Maximum input length: 1000 tokens (NER, translation)
- Summary length: 40-150 tokens
- Translation truncation: Enabled for long documents

## Limitations

- PDF and DOCX support only
- Text processing limited to first 1000 characters for some operations
- Requires internet connection for model downloads
- Arabic models require Hugging Face authentication

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
- Open an issue on GitHub
- Check the documentation for troubleshooting
- Ensure all dependencies are correctly installed

## Acknowledgments

- Hugging Face for providing pre-trained models
- Streamlit for the web application framework
- PyMuPDF for PDF processing capabilities
- CAMeL Lab for Arabic NLP models

---

**Made with ❤️ by Habiba**
