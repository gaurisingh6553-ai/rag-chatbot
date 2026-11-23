import argparse
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_PDF = PROJECT_DIR / "document_loader" / "Unit_3_Java_Questions_and_Answers.pdf"
COLLECTION_NAME = "java_unit_3"


def find_pdf(requested_path: str | None = None) -> Path:
    if requested_path:
        pdf_path = Path(requested_path).expanduser()
        if not pdf_path.is_absolute():
            pdf_path = PROJECT_DIR / pdf_path
    elif DEFAULT_PDF.exists():
        pdf_path = DEFAULT_PDF
    else:
        pdf_files = list((PROJECT_DIR / "document_loader").glob("*.pdf"))
        pdf_path = pdf_files[0] if pdf_files else PROJECT_DIR / "*.pdf"

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    return pdf_path


def build_rag(pdf_path: Path):
    print(f"Loading PDF: {pdf_path}")
    documents = PyPDFLoader(str(pdf_path)).load()
    if not documents:
        raise ValueError("The PDF contains no readable text.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    client = QdrantClient(location=":memory:")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    vectorstore.add_documents(chunks)

    print(f"Indexed {len(documents)} PDF pages into {len(chunks)} chunks.")
    return vectorstore.as_retriever(search_kwargs={"k": 4}), chunks


def build_llm():
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        return None

    try:
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
    except ImportError:
        print("NVIDIA package is not installed; using context-only answers.")
        return None

    return ChatNVIDIA(
        model="nvidia/nemotron-3-ultra-550b-a55b",
        nvidia_api_key=api_key,
        temperature=0.2,
        max_completion_tokens=1500,
    )


def make_answer(question: str, retriever) -> tuple[str, list]:
    retrieved = retriever.invoke(question)
    context = "\n\n".join(
        f"[Source: page {doc.metadata.get('page', 0) + 1}]\n{doc.page_content.strip()}"
        for doc in retrieved
    )

    llm = build_llm()
    if llm is not None:
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a Java tutor. Answer strictly from the supplied PDF context. "
                "Be clear and concise. If the answer is not in the context, say "
                "'The answer was not found in the uploaded PDF.'",
            ),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ])
        answer = (prompt | llm | StrOutputParser()).invoke(
            {"context": context, "question": question}
        )
    else:
        answer = (
            "No NVIDIA API key is configured, so here is the most relevant context "
            "retrieved from the PDF:\n\n" + context
        )

    return answer, retrieved


def print_result(question: str, answer: str, retrieved: list) -> None:
    print("\n" + "=" * 72)
    print(f"QUESTION: {question}")
    print("=" * 72)
    print(answer)
    print("\nRetrieved sources:")
    for index, doc in enumerate(retrieved, start=1):
        page = doc.metadata.get("page", 0) + 1
        preview = " ".join(doc.page_content.split())[:180]
        print(f"  [{index}] Page {page}: {preview}...")


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Ask questions about an uploaded PDF")
    parser.add_argument("pdf", nargs="?", help="Optional path to a PDF")
    parser.add_argument("question", nargs="?", help="Optional question; omit for interactive mode")
    parser.add_argument("--test", action="store_true", help="Run a built-in retrieval test")
    args = parser.parse_args()

    try:
        pdf_path = find_pdf(args.pdf)
        retriever, _ = build_rag(pdf_path)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.test:
        question = "What is Java and list its main features?"
        answer, retrieved = make_answer(question, retriever)
        print_result(question, answer, retrieved)
        return 0

    if args.question:
        answer, retrieved = make_answer(args.question, retriever)
        print_result(args.question, answer, retrieved)
        return 0

    print("\nEnter a question (or 'exit' to quit).")
    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        answer, retrieved = make_answer(question, retriever)
        print_result(question, answer, retrieved)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
