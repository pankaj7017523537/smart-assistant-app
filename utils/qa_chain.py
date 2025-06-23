from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from transformers import pipeline

# ✅ Use Roberta QA pipeline directly
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

def create_vector_store(text: str):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.create_documents([text])
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

def answer_question(vectorstore, question, _model=None):
    docs = vectorstore.similarity_search(question)
    if not docs:
        return "❌ No relevant context found.", "No document sections matched the question."

    # Use the top matching document
    context = docs[0].page_content

    # Run QA inference directly
    result = qa_pipeline(question=question, context=context)
    answer = result.get("answer", "No answer found.")
    justification = context[:300] + "..."

    return answer, justification
