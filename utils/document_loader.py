from PyPDF2 import PdfReader

def load_pdf(filepath: str) -> str:
    """
    Extracts and returns text content from a PDF file.

    Args:
        filepath (str): Path to the PDF file.

    Returns:
        str: Extracted text content.
    """
    try:
        reader = PdfReader(filepath)
        text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
        return text.strip() if text else "❌ No readable text found in PDF."
    except Exception as e:
        return f"❌ Failed to read PDF: {e}"

def load_txt(filepath: str) -> str:
    """
    Loads and returns text content from a .txt file.

    Args:
        filepath (str): Path to the text file.

    Returns:
        str: File content.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read().strip()
        return content if content else "❌ The text file is empty."
    except Exception as e:
        return f"❌ Failed to read text file: {e}"
