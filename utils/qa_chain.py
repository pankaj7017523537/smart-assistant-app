from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from transformers import pipeline

# ✅ Load question-answering pipeline (using PyTorch backend)
qa_pipeline = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2",
    tokenizer="deepset/roberta-base-squad2",
    framework="pt"
)

def create_vector_store(text: str):
    """
    Splits text into chunks, generates embeddings, and creates a FAISS vector store.
    """
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.create_documents([text])

    # Updated import location for HuggingFaceEmbeddings in LangChain >= 0.2.2
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create vector store using FAISS
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

def answer_question(vectorstore, question: str, _model=None):
    """
    Finds the most relevant document chunk using vector search, then applies QA model.
    """
    try:
        docs = vectorstore.similarity_search(question, k=1)
        if not docs:
            return "❌ No relevant context found.", "No document sections matched the question."

        context = docs[0].page_content

        result = qa_pipeline(question=question, context=context)
        answer = result.get("answer", "No answer found.")
        justification = context[:300] + "..." if context else "No context available."

        return answer.strip(), justification.strip()
    
    except Exception as e:
        return f"❌ Failed to answer question: {e}", "Error occurred during QA processing."
