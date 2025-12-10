import streamlit as st
import faiss
import numpy as np
import tempfile

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

# -------------------------------
# -------- SETTINGS ------------
# -------------------------------
EMBEDDING_MODEL = "qwen3-embedding:0.6b"  # embedding model
LLM_MODEL = "qwen2.5:3b"                 # LLM for generation
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3

# -------------------------------
# -------- UI SETUP ------------
# -------------------------------
st.set_page_config(page_title="📄 RAG Chatbot (Qwen3 Embeddings + Qwen LLM)", layout="wide")
st.title("📄 RAG Chatbot (Qwen3 Embeddings + Qwen LLM)")

st.markdown(
    "Upload PDFs or text documents, embed them with Qwen3-embedding:0.6b, "
    "and ask questions answered by a Qwen model."
)

with st.sidebar:
    st.header("Instructions")
    st.write(
        """
1. Upload PDFs or TXT files.  
2. Pipeline: Load → Chunk → Embed → Query.  
3. Ask a question.  
4. Retrieved chunks display similarity scores.
        """
    )

# -------------------------------
# -------- GLOBALS -------------
# -------------------------------
index = None
documents = []

# Initialize embedding model
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

# -------------------------------
# -------- 1. UPLOAD -----------
# -------------------------------
st.header("Step 1: Upload Documents")
uploaded_files = st.file_uploader("Upload PDFs or TXT", type=["pdf", "txt"], accept_multiple_files=True)

if uploaded_files:
    st.subheader("📥 Loading Documents")
    all_text = ""
    progress_text = st.empty()
    progress_bar = st.progress(0)

    for i, file in enumerate(uploaded_files, start=1):
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            temp.write(file.read())
            temp_path = temp.name

        loader = PyMuPDFLoader(temp_path)
        docs = loader.load()
        text = " ".join([d.page_content for d in docs])
        all_text += text + "\n"

        progress_text.text(f"Loaded {i}/{len(uploaded_files)} files...")
        progress_bar.progress(i / len(uploaded_files))

    st.success(f"✅ Loaded {len(uploaded_files)} documents.")

    # -------------------------------
    # -------- 2. CHUNKING ----------
    # -------------------------------
    st.header("Step 2: Chunking Documents")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    documents = splitter.split_text(all_text)
    st.info(f"Text split into {len(documents)} chunks.")

    # -------------------------------
    # -------- 3. EMBEDDING & INDEX -
    # -------------------------------
    st.header("Step 3: Embedding & FAISS Index")
    with st.spinner("Computing embeddings and building FAISS index..."):
        vectors = embeddings.embed_documents(documents)
        vectors_np = np.array(vectors, dtype="float32")

        dim = vectors_np.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(vectors_np)

    st.success("✅ Embeddings created and FAISS index built!")

# -------------------------------
# -------- 4. QUERY / CHAT ------
# -------------------------------
if index is not None:
    st.header("Step 4: Ask a Question")
    query = st.text_input("Your question:")

    if query:
        with st.spinner("Retrieving chunks and generating answer..."):
            # Embed query
            q_vec = embeddings.embed_query(query)
            q_vec_np = np.array([q_vec], dtype="float32")

            # Search in FAISS
            distances, indices = index.search(q_vec_np, TOP_K)

            # Prepare retrieved chunks with similarity scores
            retrieved = []
            for i, dist in zip(indices[0], distances[0]):
                score = 1 / (1 + dist)  # L2 distance -> similarity
                retrieved.append((documents[i], score))

            # Display retrieved chunks
            st.subheader("Retrieved Chunks (with similarity scores)")
            for idx, (chunk, score) in enumerate(retrieved, start=1):
                with st.expander(f"Chunk {idx} — Similarity: {score:.3f}"):
                    st.write(chunk)

            # Build context for LLM
            context = "\n\n".join([chunk for chunk, _ in retrieved])
            prompt = f"""
You are a helpful assistant.

Use ONLY the information from the context to answer the question.

### Context:
{context}

### Question:
{query}

### Answer:"""

            # Generate answer using Qwen LLM
            llm = Ollama(model=LLM_MODEL)
            answer = llm.invoke(prompt)

        # Display answer
        st.subheader("Answer")
        st.info(answer)
