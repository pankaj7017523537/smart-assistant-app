### 📚 Smart Assistant for Research Summarization
🎯 Project Objective
A GenAI-powered assistant that can:
- Generate concise summaries (≤150 words)
- Answer contextual questions from uploaded PDFs or TXT files
- Challenge users with logic-based questions and evaluate responses
- Highlight source-based justifications
- Generate downloadable PDF reports

---

###🏗️ Architecture / Reasoning Flow

[User Uploads PDF/TXT] 
        ↓
 [Text Extraction (PyMuPDF/Plaintext)]
        ↓
 [Text Summarization using Gemini-2.0]
        ↓
 [Chunked Text Embedding → FAISS Index]
        ↓
 [User Q → Similarity Search]
        ↓
 [FLAN-T5 Model → Final Answer + Justification]
```
###🗂️ Project Structure
smart-assistant-app/
│
├── app.py                     # Streamlit application
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── .gitignore                 # Git ignores venv & temp
│
└── utils/
    ├── document_loader.py     # Text/PDF loader
    ├── qa_chain.py            # QA vector search pipeline
    ├── summarizer.py          # Gemini summary generation
    └── challenge_mode.py      # MCQ challenge logic
```
###🧠 Technologies Used

Streamlit

Google Gemini (via google.generativeai)

HuggingFace Transformers: flan-t5-base, sentence-transformers

LangChain + FAISS

PyMuPDF

FPDF (for report generation)
```
### Create & Activate Virtual Environment
python -m venv venv
venv\Scripts\activate     # Windows
# or
source venv/bin/activate  # macOS/Linux

```
###📌 Other Projects

🔹 Asthma Disease Prediction & Suggestion System (Streamlit App)
    Predicts asthma likelihood based on lifestyle and environment features.

   Suggests improvements and supports bilingual UI (English/Hindi), PDF reports, and dark mode.

🔹 ATM Simulation System in Java
   Console-based project simulating ATM functionalities.

   Implements OOP concepts like encapsulation, abstraction, and inheritance.


## 🚀 Setup Instructions

### 1. Clone the repository:
```bash
git clone https://github.com/pankaj7017523537/smart-assistant-app.git
cd smart-assistant-app
Project link--> https://github.com/pankaj7017523537/Asthma-prediction-app
Project link--> https://github.com/pankaj7017523537/ATM-Simulation
```
Create a Virtual Environment

python -m venv venv
venv\Scripts\activate  # On Windows
# OR
source venv/bin/activate  # On Mac/Linux

##Run the Streamlit App
streamlit run app.py
