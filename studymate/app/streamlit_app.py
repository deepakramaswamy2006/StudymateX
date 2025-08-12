import sys
import os

# Ensure the project root is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
import sys
import os

# Add project root to Python path so sibling modules can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import tempfile
from pathlib import Path
import os
from dotenv import load_dotenv

from ingest.pdf_ingest import process_uploaded_files
from ingest.chunker import make_chunks
from embedding.embed_index import EmbeddingIndex
from retriever.retriever import Retriever
from llm.groq_client import GroqLLM
from utils.io_utils import save_session_log

load_dotenv()

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE_WORDS", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP_WORDS", 100))
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH", 32))
TOP_K = int(os.getenv("TOP_K", 3))

st.set_page_config(page_title="StudyMate - Llama 3.1 8B", layout="wide")
st.title("📚 StudyMate — PDF Q&A (Groq Llama 3.1 8B Instant)")

if "index" not in st.session_state:
    st.session_state.index = None
if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    tmpdir = Path(tempfile.gettempdir()) / "studymate_uploads"
    tmpdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for uf in uploaded_files:
        p = tmpdir / uf.name
        p.write_bytes(uf.getbuffer())
        paths.append(p)

    corpus = process_uploaded_files(paths)

    metadatas, texts = [], []
    for doc in corpus:
        chunks = make_chunks(doc["text"], chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        for c in chunks:
            metadatas.append({"source": doc["path"], "chunk_id": c["chunk_id"], "text": c["text"]})
            texts.append(c["text"])

    emb_index = EmbeddingIndex(model_name=EMBED_MODEL, normalize=True)
    embeddings = emb_index.embed_texts(texts, batch_size=BATCH_SIZE)
    emb_index.build_index(embeddings, metadatas)
    st.session_state.index = emb_index
    st.success("✅ PDF processed and indexed.")

# Display previous messages (old messages appear above new ones)
for msg in st.session_state.messages:
    with st.chat_message(msg.get("role", "assistant")):
        st.markdown(msg.get("content", ""))

# Chat input at the bottom
question = st.chat_input("Ask a question")
if question:
    if st.session_state.index is None:
        st.warning("Please upload and process PDFs first.")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                retriever = Retriever(st.session_state.index)
                hits = retriever.retrieve(question, top_k=TOP_K)

                context = "\n\n".join([
                    f"Source: {h['meta']['source']} (chunk {h['meta']['chunk_id']})\n{h['meta']['text']}"
                    for h in hits
                ])
                prompt = (
                    "Answer the question using only the following context:\n\n"
                    f"{context}\n\nQuestion: {question}"
                )
                try:
                    llm = GroqLLM()
                    answer = llm.generate(prompt)
                except Exception as e:
                    answer = f"Error generating answer: {e}"
            st.markdown(answer)
        # Save assistant reply
        st.session_state.messages.append({"role": "assistant", "content": answer})

