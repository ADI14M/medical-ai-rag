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

st.markdown(
    """
    <style>
    /* Make the title sticky */
    div[data-testid="stVerticalBlock"] > div:first-child {
        position: sticky;
        top: 2.875rem;
        z-index: 999;
        background-color: #0e1117;
        padding-top: 1rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid #333;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.title("🩺 Visukhi Medical Chatbot")

# ====================== Settings ======================
MODEL_NAME = "phi3"
TEMPERATURE = 0.2

# Increased limits to fetch and process far more patients per query
TOP_K = 40
FETCH_K = 100
MAX_CONTEXT_CHARS = 12000

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
        search_kwargs = {"k": TOP_K, "fetch_k": FETCH_K, "lambda_mult": 0.5}

        retriever = st.session_state.vectorstore.as_retriever(
            search_type="mmr",
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

        # ====================== DB STATS (FOR AGGREGATE QUERIES) ======================
        if "db_stats" not in st.session_state:
            try:
                import psycopg2
                from config import DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT
                conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD, port=DB_PORT)
                cur = conn.cursor()
                
                cur.execute("SELECT COUNT(*) FROM oads.patients")
                total_patients = cur.fetchone()[0]
                
                cur.execute("SELECT COUNT(*) FROM oads.analysis")
                total_analyses = cur.fetchone()[0]
                
                cur.execute("SELECT COUNT(*) FROM oads.studies")
                total_studies = cur.fetchone()[0]
                
                conn.close()
                
                st.session_state.db_stats = {
                    "total_patients": total_patients,
                    "total_analyses": total_analyses,
                    "total_studies": total_studies
                }
            except Exception:
                st.session_state.db_stats = {
                    "total_patients": "Unknown",
                    "total_analyses": "Unknown",
                    "total_studies": "Unknown"
                }

        stats = st.session_state.db_stats

        # ====================== PROMPT ======================
        prompt_template = ChatPromptTemplate.from_template("""
You are a strict medical assistant.

[ DATABASE STATISTICS ]
Total Registered Patients: {total_patients}
Total Medical Studies: {total_studies}
Total Image Analyses: {total_analyses}

[ PATIENT RECORDS CONTEXT ]
{context}

[ RULES ]
1. If the user asks for total counts, numbers, or aggregates, you MUST answer using the DATABASE STATISTICS.
2. If the user asks about specific patient details, you MUST answer using the PATIENT RECORDS CONTEXT.
3. If the answer cannot be found in either section, exactly output: "Not found in database"
4. Do not guess or make up information.

Question: {question}

Answer:
""")

        final_prompt = prompt_template.format(
            total_patients=stats["total_patients"],
            total_studies=stats["total_studies"],
            total_analyses=stats["total_analyses"],
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

        # ====================== TIME ======================
        end_time = time.time()
        st.caption(f"⏱️ {(end_time - start_time):.2f} sec")

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response
    })