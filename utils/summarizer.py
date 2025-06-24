from transformers import pipeline

# Load the local summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def generate_summary(text: str, max_len: int = 1300) -> str:
    """
    Generates a summary from the input text, breaking it into chunks if too long.

    Args:
        text (str): The input text to summarize.
        max_len (int): Total number of characters to process (approximate limit).

    Returns:
        str: Summarized version of the text.
    """
    try:
        # Short documents can be summarized directly
        if len(text) <= 1500:
            result = summarizer(
                text,
                max_length=200,
                min_length=50,
                do_sample=False
            )[0]['summary_text']
            return result.strip()

        # For longer texts, chunk them
        chunks = [text[i:i+1000] for i in range(0, min(len(text), max_len), 1000)]
        summaries = []
        for i, chunk in enumerate(chunks[:3]):  # Limit to first 3 chunks
            result = summarizer(
                chunk,
                max_length=200,
                min_length=50,
                do_sample=False
            )[0]['summary_text']
            summaries.append(result.strip())

        return "\n".join(summaries)
    
    except Exception as e:
        return f"❌ Failed to generate summary: {e}"
