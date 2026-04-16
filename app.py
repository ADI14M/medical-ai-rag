import streamlit as st
from langchain_community.vectorstores import FAISS
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
        background-color: #c6cbd3;
        padding-top: 1rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid #a0a6b1;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.title("🩺 Visukhi Medical Chatbot")

# ====================== Settings ======================
MODEL_NAME = "tinyllama"
TEMPERATURE = 0.2


TOP_K = 8
FETCH_K = 30
MAX_CONTEXT_CHARS = 3000


@st.cache_resource
def load_vectorstore():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    try:
        return FAISS.load_local("./faiss_db", embeddings)
    except Exception as e:
        st.error(f"Failed to load FAISS DB. Run embed.py first. Error: {e}")
        return None


if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    with st.spinner("Loading vector database..."):
        st.session_state.vectorstore = load_vectorstore()



for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


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
                
                cur.execute("SELECT COUNT(*) FROM ehr.patients")
                total_patients = cur.fetchone()[0]
                
                cur.execute("SELECT COUNT(*) FROM ehr.labevents")
                total_labevents = cur.fetchone()[0]
                
                conn.close()
                
                st.session_state.db_stats = {
                    "total_patients": total_patients,
                    "total_labevents": total_labevents
                }
            except Exception:
                st.session_state.db_stats = {
                    "total_patients": "Unknown",
                    "total_labevents": "Unknown"
                }

        stats = st.session_state.db_stats

        # ====================== PROMPT ======================
        prompt_template = ChatPromptTemplate.from_template("""
You are a direct, robotic medical assistant.

[ DATABASE STATISTICS ]
Total Registered Patients: {total_patients}
Total EHR Lab Events: {total_labevents}

[ PATIENT LAB EVENT CONTEXT ]
{context}

[ STRICT RULES ]
1. Answer using ONLY natural language. NEVER output raw SQL queries or database code.
2. Focus on clinical summarization, specifically pointing out any values flagged as ABNORMAL.
3. Provide cohesive patient insights based on the retrieved lab events.
4. Output ONLY the final analytical answer. DO NOT explain your reasoning.
5. If the answer cannot be confidently deduced from the Context or Database Statistics, output exactly: "Not found in database".

Question: {question}

Final Clinical Answer:
""")

        final_prompt = prompt_template.format(
            total_patients=stats["total_patients"],
            total_labevents=stats["total_labevents"],
            context=context,
            question=prompt
        )

        # ====================== LLM ======================
        llm = ChatOllama(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            num_ctx=2048  # Hard cap memory allocation to stay under 2.5 GiB
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