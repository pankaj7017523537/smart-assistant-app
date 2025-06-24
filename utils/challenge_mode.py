from transformers import pipeline
from sentence_transformers import CrossEncoder
import google.generativeai as genai

# ✅ Text generator for questions (simple GPT-2 model)
text_gen = pipeline("text-generation", model="gpt2")

# ✅ Load semantic evaluator model once
evaluator = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ✅ Gemini model setup (used for answer evaluation only)
try:
    gemini_model = genai.GenerativeModel("gemini-2.0-flash")
except Exception:
    gemini_model = None  # Fallback if not available


def generate_questions(text: str):
    """
    Generates 3 logic/comprehension questions using GPT-2.
    """
    prompt = (
        f"Generate 3 unique comprehension or logic-based questions "
        f"based on the following document:\n\n{text[:1000]}\n\nQuestions:"
    )
    try:
        generated = text_gen(prompt, max_new_tokens=256, do_sample=True)[0]["generated_text"]
        lines = generated.split('\n')
        questions = [line.strip() for line in lines if line.strip().endswith('?')]
        return questions[:3] if questions else [
            "What is the main goal of the document?",
            "Explain one method used in the paper.",
            "Summarize the conclusion in one sentence."
        ]
    except Exception as e:
        return [f"❌ Error generating questions: {e}"]


def evaluate_answer(question: str, answer: str, text: str, _=None):
    """
    Evaluates user answer using Gemini (if available), otherwise fallback to CrossEncoder.
    """

    # 🛡️ Guard against empty answers
    if not answer.strip():
        return "❌ No answer provided."

    # ✨ Gemini-based Evaluation
    if gemini_model:
        try:
            prompt = (
                f"You're an academic evaluator.\n"
                f"Document:\n{text[:1000]}\n\n"
                f"Question: {question}\n"
                f"Student Answer: {answer}\n\n"
                f"Evaluate the quality of the student's answer and return concise feedback with a score (0–1)."
            )
            response = gemini_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"⚠️ Gemini evaluation failed: {e}"

    # 🔁 Fallback to CrossEncoder (Semantic similarity)
    try:
        reference = f"Question: {question}\nRelevant Document Part: {text[:1000]}"
        score = evaluator.predict([(reference, answer)])[0]

        if score > 0.75:
            return f"✅ Good answer. You captured it well! (Score: {score:.2f})"
        elif score > 0.5:
            return f"🟡 Partial match. Consider refining your response. (Score: {score:.2f})"
        else:
            return f"❌ Incorrect or off-topic. Please refer back to the document. (Score: {score:.2f})"
    except Exception as e:
        return f"❌ Failed to evaluate answer: {e}"
