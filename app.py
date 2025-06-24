import os
import streamlit as st
import google.generativeai as genai
from fpdf import FPDF
import warnings

# Environment setup
os.environ["STREAMLIT_WATCH_FILE_SYSTEM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)

# ✅ Gemini API Key
genai.configure(api_key="AIzaSyBq2Z5-zXkcwNKGKig5fGm5ZwKi1jkg7_I")
model = genai.GenerativeModel("gemini-2.0-flash")

# ✅ Custom utility imports
from utils.document_loader import load_pdf, load_txt
from utils.summarizer import generate_summary
from utils.qa_chain import create_vector_store, answer_question
from utils.challenge_mode import generate_questions, evaluate_answer

# ✅ Streamlit App Config
st.set_page_config(page_title="Smart Research Assistant", layout="wide")
st.title("📚 Smart Assistant for Research Summarization")

# 📌 Upload PDF or TXT
uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

if uploaded_file:
    # Save file
    os.makedirs("temp", exist_ok=True)
    filepath = os.path.join("temp", uploaded_file.name)
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load content
    text = load_pdf(filepath) if uploaded_file.name.endswith(".pdf") else load_txt(filepath)

    # Preview first 1000 characters
    st.subheader("📁 Extracted Document Preview")
    st.code(text[:1000] if text.strip() else "❌ No readable text found in document!", language="markdown")

    # Check for valid content
    if text.startswith("❌") or not text.strip():
        st.error("Document could not be read. Please upload a valid, readable PDF or TXT file.")
        st.stop()

    # 📄 Generate Summary
    with st.spinner("Generating summary..."):
        summary = generate_summary(text)
        st.subheader("📄 Document Summary")
        st.write(summary)

    # 🧠 Ask Anything
    st.subheader("🧠 Ask Anything")
    if "vectorstore" not in st.session_state:
        with st.spinner("Indexing document..."):
            st.session_state.vectorstore = create_vector_store(text)

    user_question = st.text_input("Ask a question based on the document")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if user_question:
        with st.spinner("Answering..."):
            answer, justification = answer_question(st.session_state.vectorstore, user_question, model)

            st.session_state.chat_history.append({
                "question": user_question,
                "answer": answer,
                "justification": justification
            })

            st.markdown("#### ✅ Answer")
            st.write(answer)

            if justification in text:
                st.markdown("#### 🔍 Justification Highlight")
                st.code(justification, language='markdown')
            else:
                st.markdown("#### 🔍 Justification")
                st.info(justification)

    # 🎯 Challenge Mode
    st.subheader("🎯 Challenge Me")
    if st.button("Generate Questions"):
        with st.spinner("Creating questions..."):
            questions = generate_questions(text)
            st.session_state.challenge = {
                "questions": questions,
                "answers": [""] * len(questions),
                "feedback": [""] * len(questions)
            }

    if "challenge" in st.session_state:
        questions = st.session_state.challenge.get("questions", [])
        st.markdown("##### 📘 Answer the following:")
        for i, q in enumerate(questions):
            st.session_state.challenge["answers"][i] = st.text_input(f"{q}", key=f"q{i}")

        if st.button("Evaluate Answers"):
            with st.spinner("Evaluating your answers..."):
                feedback_list = []
                for i in range(len(questions)):
                    feedback = evaluate_answer(
                        questions[i],
                        st.session_state.challenge["answers"][i],
                        text,
                        model
                    )
                    feedback_list.append(feedback)
                st.session_state.challenge["feedback"] = feedback_list

    # 📝 Feedback View (FIXED)
    if "challenge" in st.session_state:
        questions = st.session_state.challenge.get("questions", [])
        feedback = st.session_state.challenge.get("feedback", [])

        if questions and feedback:
            st.subheader("📝 Evaluation & Feedback")
            for i in range(min(len(questions), len(feedback))):
                st.markdown(f"**Q{i+1}:** {questions[i]}")
                st.markdown(f"**💡 Feedback:** {feedback[i]}")
                st.markdown("---")
        else:
            st.info("No questions or feedback available yet.")

    # 📅 PDF Report Download
    if st.button("📅 Download Report PDF"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Smart Research Assistant Report", ln=True, align='C')
        pdf.multi_cell(0, 10, txt=f"\nSummary:\n{summary}\n")

        if st.session_state.chat_history:
            for i, chat in enumerate(st.session_state.chat_history, 1):
                pdf.multi_cell(0, 10, txt=f"Q{i}: {chat['question']}\nA: {chat['answer']}\nJustification: {chat['justification']}\n")

        report_path = os.path.join("temp", "smart_assistant_report.pdf")
        pdf.output(report_path)
        with open(report_path, "rb") as f:
            st.download_button("Download PDF Report", f, file_name="Smart_Assistant_Report.pdf")
