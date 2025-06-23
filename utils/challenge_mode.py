from transformers import pipeline
from sentence_transformers import CrossEncoder

# Question generator pipeline (you can upgrade later if needed)
text_gen = pipeline("text-generation", model="gpt2")

# Load semantic evaluator model once
evaluator = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

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
    Evaluates user's answer by comparing it semantically to the context using a CrossEncoder.
    """
    # Use a snippet of the document as ground truth context
    context = text[:1000]

    # Combine question and context
    reference = f"Question: {question}\nRelevant Document Part: {context}"

    try:
        score = evaluator.predict([(reference, answer)])[0]

        # Map score to feedback
        if score > 0.75:
            return f"✅ Good answer. You captured it well! (Score: {score:.2f})"
        elif score > 0.5:
            return f"🟡 Partial match. Consider refining your response. (Score: {score:.2f})"
        else:
            return f"❌ Incorrect or off-topic. Please refer back to the document. (Score: {score:.2f})"
    except Exception as e:
        return f"❌ Failed to evaluate answer: {e}"
