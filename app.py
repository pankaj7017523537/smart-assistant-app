import os
import streamlit as st
import google.generativeai as genai
from fpdf import FPDF  # 🆕 For PDF report generation

# ✅ Configure Gemini API Key
genai.configure(api_key="AIzaSyBq2Z5-zXkcwNKGKig5fGm5ZwKi1jkg7_I")

# ✅ Initialize Gemini Model (used only in challenge mode now)
model = genai.GenerativeModel("gemini-2.0-flash")

# ✅ Utility Imports
from utils.document_loader import load_pdf, load_txt
from utils.summarizer import generate_summary
from utils.qa_chain import create_vector_store, answer_question
from utils.challenge_mode import generate_questions, evaluate_answer

# ✅ Streamlit App Setup
st.set_page_config(page_title="Smart Research Assistant", layout="wide")
st.title("📚 Smart Assistant for Research Summarization (Gemini)")

# 📎 Upload File
uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

if uploaded_file:
    # Save to temp
    os.makedirs("temp", exist_ok=True)
    filepath = os.path.join("temp", uploaded_file.name)
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load file
    text = load_pdf(filepath) if uploaded_file.name.endswith(".pdf") else load_txt(filepath)

    # 📄 Auto Summary
    with st.spinner("Generating summary..."):
        summary = generate_summary(text)
        st.subheader("📄 Document Summary")
        st.write(summary)

    # 🧠 Ask Anything Mode
    st.subheader("🧠 Ask Anything")
    if "vectorstore" not in st.session_state:
        with st.spinner("Indexing document..."):
            st.session_state.vectorstore = create_vector_store(text)

    # 🧠 Question Input
    user_question = st.text_input("Ask a question based on the document")

    # ➕ Memory Storage
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if user_question:
        with st.spinner("Answering..."):
            answer, justification = answer_question(st.session_state.vectorstore, user_question, model)

            # Store in memory
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

    # 🎯 Challenge Me Mode
    st.subheader("🎯 Challenge Me")
    if st.button("Generate Questions"):
        with st.spinner("Creating questions..."):
            questions = generate_questions(text)
            st.session_state.challenge = {
                "questions": questions,
                "answers": [""] * 3,
                "feedback": [""] * 3
            }

    if "challenge" in st.session_state:
        st.markdown("##### 📘 Answer the following:")
        for i, q in enumerate(st.session_state.challenge["questions"]):
            st.session_state.challenge["answers"][i] = st.text_input(f"{q}", key=f"q{i}")

        if st.button("Evaluate Answers"):
            with st.spinner("Evaluating your answers..."):
                st.session_state.challenge["feedback"] = []
                for i in range(3):
                    fb = evaluate_answer(
                        st.session_state.challenge["questions"][i],
                        st.session_state.challenge["answers"][i],
                        text,
                        None
                    )
                    st.session_state.challenge["feedback"].append(fb)

    # 📝 Feedback Display
    if "challenge" in st.session_state and st.session_state.challenge.get("feedback"):
        st.subheader("📝 Evaluation & Feedback")
        for i, fb in enumerate(st.session_state.challenge["feedback"]):
            st.markdown(f"**Q{i+1}:** {st.session_state.challenge['questions'][i]}")
            st.markdown(f"**💡 Feedback:** {fb}")
            st.markdown("---")

    # 📥 PDF Report Download
    if st.button("📥 Download Report PDF"):
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
