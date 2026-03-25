import streamlit as st
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import time

# ====================== Page Config ======================
st.set_page_config(
    page_title="Visukhi Medical Assistant",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Visukhi Medical Chatbot")

# ====================== Settings ======================
MODEL_NAME = "phi3"
TEMPERATURE = 0.2

TOP_K = 8              # ✅ FIXED
FETCH_K = 30           # ✅ FIXED
MAX_CONTEXT_CHARS = 4000   # ✅ FIXED

# ====================== Load Vector DB ======================
@st.cache_resource
def load_vectorstore():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    return Chroma(
        persist_directory="./vector_db",
        embedding_function=embeddings,
        collection_name="my_rag_collection"
    )

# ====================== Session ======================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    with st.spinner("Loading vector database..."):
        st.session_state.vectorstore = load_vectorstore()


# ====================== Display Chat ======================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ====================== Chat ======================
if prompt := st.chat_input("Ask a medical question..."):

    # Safety filter
    if any(w in prompt.lower() for w in ["suicide", "kill", "overdose"]):
        st.warning("⚠️ Cannot handle this query.")
        st.stop()

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        start_time = time.time()

        # ====================== DYNAMIC RETRIEVER ======================
        search_kwargs = {"k": TOP_K}

        retriever = st.session_state.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs
        )

        # ====================== RETRIEVE ======================
        docs = retriever.invoke(prompt)

        # ====================== CONTEXT ======================
        context = ""
        used_docs = []

        for doc in docs:
            if len(context) + len(doc.page_content) < MAX_CONTEXT_CHARS:
                context += doc.page_content + "\n\n"
                used_docs.append(doc)
            else:
                break

        # ====================== PROMPT ======================
        prompt_template = ChatPromptTemplate.from_template("""
You are a strict medical assistant.

Rules:
- Use ONLY given context
- No guessing
- If missing → say "Not found in database"

Context:
{context}

Question: {question}

Answer:
""")

        final_prompt = prompt_template.format(
            context=context,
            question=prompt
        )

        # ====================== LLM ======================
        llm = ChatOllama(
            model=MODEL_NAME,
            temperature=TEMPERATURE
        )

        stream = llm.stream(final_prompt)

        for chunk in stream:
            if hasattr(chunk, "content"):
                full_response += chunk.content
            else:
                full_response += str(chunk)

            placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)

        # ====================== SOURCES (FIXED) ======================
        st.markdown("### 📚 Patient Sources")

        patient_ids = list(set([
            doc.metadata.get("patient_id", "Unknown")
            for doc in used_docs
        ]))

        for pid in patient_ids:
            st.write(f"Patient ID: {pid}")

        # ====================== TIME ======================
        end_time = time.time()
        st.caption(f"⏱️ {(end_time - start_time):.2f} sec")

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response
    })