from transformers import pipeline

# Load summarization model once
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def generate_summary(text: str) -> str:
    """
    Summarizes the given document text using HuggingFace Transformers.

    Parameters:
        text (str): Full text from the uploaded document.

    Returns:
        str: A concise summary (≤ 150 words).
    """
    if not text.strip():
        return "No content found in document."

    # Truncate input to first 1024 characters (safe length for BART)
    input_text = text[:1024]

    try:
        result = summarizer(
            input_text,
            max_length=150,
            min_length=50,
            do_sample=False,
            truncation=True  # Ensures length limit is respected
        )
        return result[0]['summary_text'].strip()
    except Exception as e:
        return f"❌ Failed to generate summary: {e}"
