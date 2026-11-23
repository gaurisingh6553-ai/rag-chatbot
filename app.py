import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

st.set_page_config(page_title="RAG Project", page_icon="📚")
st.title("📚 My RAG Chatbot")
st.write("Upload PDFs, then ask questions from them!")

uploaded_files = st.file_uploader("Upload PDFs here", type="pdf", accept_multiple_files=True)

if uploaded_files:
    if "vectorstore" not in st.session_state:
        with st.spinner("Reading PDFs..."):
            all_docs = []
            for uploaded_file in uploaded_files:
                with open(f"temp_{uploaded_file.name}", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                loader = PyPDFLoader(f"temp_{uploaded_file.name}")
                all_docs.extend(loader.load())
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(all_docs)
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(chunks, embeddings)
            st.session_state.vectorstore = vectorstore
            st.session_state.messages = []
            st.success(f"Done! {len(all_docs)} pages, {len(chunks)} chunks created.")

    # Purani chat dikhao
    for msg in st.session_state.get("messages", []):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Naya sawaal
    if question := st.chat_input("Ask your question..."):
        with st.chat_message("user"):
            st.write(question)
        st.session_state.messages.append({"role": "user", "content": question})

        retriever = st.session_state.vectorstore.as_retriever()
        llm = ChatNVIDIA(model="nvidia/nemotron-3-ultra-550b-a55b")
        prompt = ChatPromptTemplate.from_template(
            "Answer in English, only using the context below. If not found, say 'Not found in PDF'.\n\nContext:\n{context}\n\nQuestion: {question}"
        )
        def format_docs(docs):
            return "\n\n".join(d.page_content for d in docs)
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt | llm | StrOutputParser()
        )
        with st.spinner("Finding answer..."):
            answer = chain.invoke(question)
            with st.chat_message("assistant"):
                st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})