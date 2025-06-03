import os
import streamlit as st
import chromadb
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from PyPDF2 import PdfReader
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv

# --- Load API Key securely ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Initialize ChatGroq ---
groq_chatbot = ChatGroq(api_key=GROQ_API_KEY, model_name="llama3-70b-8192")

# --- Streamlit Page Settings ---
st.set_page_config(page_title="GenAI Chatbot", layout="wide")
st.title("📚 GenAI Chatbot with RAG 🔍")

# --- File Upload ---
uploaded_file = st.file_uploader("📂 Upload a PDF file", type=["pdf"])

if uploaded_file:

    def load_pdf(file):
        try:
            reader = PdfReader(file)
            return "".join([page.extract_text() or "" for page in reader.pages])
        except Exception as e:
            st.error(f"⚠️ Error reading PDF: {str(e)}")
            return ""

    def chunk_text(text):
        chunk_size = 500 if len(text) < 10000 else (1000 if len(text) < 50000 else 2000)
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=100)
        return splitter.split_text(text)

    pdf_text = load_pdf(uploaded_file)

    if pdf_text.strip():
        chunks = chunk_text(pdf_text)
        st.success(f"✅ PDF processed into {len(chunks)} chunks.")

        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        collection = chroma_client.get_or_create_collection(name="ai_knowledge_base")

        existing_docs = set(collection.get().get("documents", []))
        new_chunks = [chunk for chunk in chunks if chunk not in existing_docs]

        if new_chunks:
            embeddings = [embedding_model.embed_query(chunk) for chunk in new_chunks]
            collection.add(
                ids=[str(i) for i in range(len(existing_docs), len(existing_docs) + len(new_chunks))],
                documents=new_chunks,
                embeddings=embeddings
            )
            st.success("✅ New embeddings stored in ChromaDB!")
        else:
            st.success(f"✅ Retrieved {len(existing_docs)} embeddings from ChromaDB!")
    else:
        st.warning("⚠️ No text extracted from PDF. Please check your file.")

    def retrieve_context(query, top_k=3):
        query_embedding = embedding_model.embed_query(query)
        results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
        return results.get("documents", [[]])[0] if results else "No relevant context found."

    def query_llama3(context, user_query):
        system_prompt = """
        You are an advanced AI chatbot with Retrieval-Augmented Generation (RAG) capabilities. Retrieve relevant knowledge before generating responses.
        """
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Context: {context}\n\nQuestion: {user_query}")
        ]
        response = groq_chatbot(messages)
        return response.content.strip()

    def evaluate_answer(question, context, answer):
        eval_prompt = f"""
        Evaluate the given answer based on the provided context and question.

        Question: {question}
        Context: {context}
        Answer: {answer}

        Provide a score from 0 to 10 and brief feedback.
        """
        messages = [
            SystemMessage(content="You are an AI evaluating chatbot responses."),
            HumanMessage(content=eval_prompt)
        ]
        response = groq_chatbot(messages)
        return response.content.strip()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.subheader("💬 Ask a Question")
    user_query = st.text_input("Type your question here...")

    if st.button("🔍 Get Answer") and user_query:
        with st.spinner("Retrieving knowledge..."):
            retrieved_context = retrieve_context(user_query)
            st.write("📚 **Retrieved Context:**")
            st.info(retrieved_context)

            response = query_llama3(retrieved_context, user_query)
            st.subheader("🤖 Answer")
            st.success(response)

            eval_result = evaluate_answer(user_query, retrieved_context, response)
            st.subheader("📊 Evaluation")
            st.warning(eval_result)

            st.session_state.chat_history.append({
                "question": user_query,
                "answer": response,
                "evaluation": eval_result
            })

    if st.session_state.chat_history:
        st.subheader("🐜 Chat History")
        for chat in st.session_state.chat_history[::-1]:
            with st.expander(f"**Q:** {chat['question']}"):
                st.write(f"**A:** {chat['answer']}")
                st.write(f"**Evaluation:** {chat['evaluation']}")
