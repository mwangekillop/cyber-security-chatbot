import fitz  # PyMuPDF - for PDF text + page numbers
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, pipeline
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

BOOKS_FOLDER = "books"
st.set_page_config(page_title="Computer Science Chatbot", layout="wide")
st.title("📚 Computer Science Chatbot - Cyber Security")
st.write("Answers only from your PDF textbooks with page numbers.")

# --- 1. TRANSFORMER FUNDAMENTALS IMPLEMENTED ---
# Tokenization: We use AutoTokenizer to break text into tokens
# Token Embeddings: SentenceTransformer converts tokens into vectors
# Self-attention: Transformer looks at all tokens in context to understand meaning
# Pretrained models: We use all-MiniLM-L6-v2 (embedding) and flan-t5-small (generative)
# Embedding vs Generative: Embedding finds meaning (search), Generative writes answer

@st.cache_resource
def load_models():
    # A. EMBEDDING TRANSFORMER - converts text to semantic vectors for search
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    # Tokenizer to show tokenization concept
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    # B. GENERATIVE TRANSFORMER - uses retrieved passages to formulate answer
    # Using small model for speed on laptop
    try:
        generator = pipeline('text2text-generation', model='google/flan-t5-small')
    except:
        generator = None
    return embedding_model, tokenizer, generator

@st.cache_data
def load_books():
    docs = []
    if not os.path.exists(BOOKS_FOLDER):
        return docs
    for file in os.listdir(BOOKS_FOLDER):
        if file.lower().endswith(".pdf"):
            path = os.path.join(BOOKS_FOLDER, file)
            pdf = fitz.open(path)
            for page_num in range(len(pdf)):
                text = pdf[page_num].get_text().strip()
                if len(text) > 100:
                    for i in range(0, len(text), 1000):
                        chunk = text[i:i+1000]
                        if len(chunk) > 100:
                            docs.append({"text": chunk, "source": file, "page": page_num + 1})
    return docs

embedding_model, tokenizer, generator = load_models()
all_docs = load_books()

if not all_docs:
    st.error(f"No books found in {BOOKS_FOLDER}/ folder!")
else:
    st.success(f"Loaded {len(all_docs)} chunks from {len(os.listdir(BOOKS_FOLDER))} books using PyMuPDF!")

    # --- OPTIMIZED EMBEDDING LOADING (fixes your forever loading) ---
    EMB_FILE = "embeddings.npy"
    def get_embeddings(_docs):
        texts = [d["text"] for d in _docs]
        # If saved embeddings exist, load them (10 seconds)
        if os.path.exists(EMB_FILE):
            return np.load(EMB_FILE)
        # First time: create and save (slow once)
        embs = embedding_model.encode(texts, show_progress_bar=True)
        np.save(EMB_FILE, embs)
        return embs

    embeddings = get_embeddings(all_docs)

    # Show tokenization example for deliverable
    with st.expander("See Transformer Concepts (for report)"):
        sample = "What is cyber security?"
        tokens = tokenizer.tokenize(sample)
        st.write(f"*Tokenization Example:* '{sample}' -> Tokens: {tokens}")
        st.write("*Self-attention:* Model weighs all tokens to understand 'cyber' relates to 'security'")
        st.write("*Embedding Model:* all-MiniLM-L6-v2 (384-dim vectors for semantic search)")
        st.write("*Generative Model:* google/flan-t5-small (formulates answer from retrieved passages)")

    question = st.text_input("Ask a question from your textbooks:")
    if question:
        q_emb = embedding_model.encode([question])
        sims = cosine_similarity(q_emb, embeddings)[0]
        top_idx = np.argsort(sims)[-3:][::-1]

        # Retrieve relevant passages
        retrieved = []
        sources = []
        for idx in top_idx:
            if sims[idx] > 0.25:
                retrieved.append(all_docs[idx])
                sources.append(all_docs[idx])

        if retrieved:
            # GENERATIVE STEP: Use retrieved passages to generate answer
            context = " ".join([d["text"][:500] for d in retrieved])
            if generator:
                prompt = f"Answer the question using only this context: {context} Question: {question}"
                try:
                    gen_answer = generator(prompt, max_length=200)[0]['generated_text']
                except:
                    gen_answer = "Based on your textbooks:"
            else:
                gen_answer = "Based on your textbooks (Generative model loading on first run):"

            st.subheader("Answer from your books:")
            st.write(gen_answer)
            st.divider()
            st.write("*Sources & Citations:*")
            for doc in sources:
                st.write(f"{doc['source']} - Page {doc['page']}")
                st.write(doc["text"][:500] + "...")
                st.divider()
        else:
            st.warning("No relevant passage found in books. I will not guess (as required).")

# Show required softwares
with st.sidebar:
    st.header("Softwares Used")
    st.write("1. Python\n2. VS Code\n3. Git\n4. PyMuPDF\n5. Sentence Transformers (Embedding Transformer)\n6. Streamlit\n7. Transformers (Generative Transformer)\n8. PyTorch\n9. scikit-learn")
    st.write("*Base model compatible for integration: all-MiniLM-L6-v2*")