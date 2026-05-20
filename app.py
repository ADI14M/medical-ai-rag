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



import streamlit.components.v1 as components

components.html(
    """
    <script>
    const doc = window.parent.document;
    const oldBtn = doc.getElementById('chatgpt-scroll-btn');
    if (oldBtn) oldBtn.remove();

    let btn = doc.createElement('div');
    btn.id = 'chatgpt-scroll-btn';
    btn.innerHTML = '&#8595;'; 
    btn.style.cssText = "position:fixed; bottom:100px; left:50%; transform:translateX(-50%); width:36px; height:36px; border-radius:50%; background-color:#ffffff; color:#333333; text-align:center; line-height:34px; font-size:20px; cursor:pointer; z-index:999999; box-shadow:0px 2px 8px rgba(0,0,0,0.15); border:1px solid #e5e5e5; display:none;";
    doc.body.appendChild(btn);

    btn.addEventListener('click', () => {
        const msgs = doc.querySelectorAll('[data-testid=\"stChatMessage\"]');
        if (msgs.length > 0) {
            msgs[msgs.length - 1].scrollIntoView({ behavior: 'smooth', block: 'end' });
        }
    });

    setInterval(() => {
        const msgs = doc.querySelectorAll('[data-testid=\"stChatMessage\"]');
        if(msgs.length > 0) {
            const lastMsg = msgs[msgs.length - 1];
            const rect = lastMsg.getBoundingClientRect();
            if (rect.bottom > window.parent.innerHeight + 50) {
                btn.style.display = 'block';
            } else {
                btn.style.display = 'none';
            }
        }
    }, 300);
    </script>
    """,
    height=0
)

st.markdown(
    """
    <style>
    /* Make the title sticky */
    div[data-testid="stHeadingWithActionElements"] {
        position: sticky;
        top: 2.875rem;
        z-index: 999;
        background-color: #c6cbd3;
        padding-top: 1rem;
        border-bottom: 1px solid #a0a6b1;
    }
    /* Stop chat text from overlapping behind the sticky input bar */
    div.block-container {
        padding-bottom: 150px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.title("🩺 Visukhi Medical Chatbot")

# ====================== Patient Search Sidebar ======================
@st.cache_data(ttl=3600)
def get_all_patient_names():
    try:
        import psycopg2
        from config import DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT
        conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD, port=DB_PORT)
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT full_name FROM oads.patients ORDER BY full_name")
        names = [row[0] for row in cur.fetchall() if row[0]]
        conn.close()
        return names
    except Exception as e:
        return []

with st.sidebar:
    st.header("🔍 Patient Search")
    
    patient_names = get_all_patient_names()
    options = [""] + patient_names
    
    search_name = st.selectbox(
        "Search and Select a Patient:",
        options=options,
        index=0,
        help="Start typing a name to see suggestions"
    )
    
    if search_name:
        with st.spinner(f"Searching for {search_name}..."):
            try:
                import psycopg2
                from config import DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT
                conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD, port=DB_PORT)
                cur = conn.cursor()
                
                # Fetch patient summary from OADS
                query = """
                SELECT p.patient_id, p.full_name, p.gender, s.study_date, s.priority, i.image_type, a.findings_summary, a.confidence_score
                FROM oads.patients p
                LEFT JOIN oads.studies s ON p.patient_id = s.patient_id
                LEFT JOIN oads.images i ON s.study_id = i.study_id
                LEFT JOIN oads.analysis a ON i.image_id = a.image_id
                WHERE p.full_name ILIKE %s
                ORDER BY s.study_date DESC
                LIMIT 10;
                """
                cur.execute(query, (f"%{search_name}%",))
                rows = cur.fetchall()
                conn.close()
                
                if rows:
                    patient_info = rows[0]
                    st.subheader(f"Patient: {patient_info[1]}")
                    st.write(f"**ID:** {patient_info[0]} | **Gender:** {patient_info[2].capitalize() if patient_info[2] else 'Unknown'}")
                    
                    if patient_info[3]: # Has studies
                        st.markdown("### Recent Studies & Findings")
                        studies_text = ""
                        for r in rows:
                            if r[3]: # study_date
                                conf_str = f"(Conf: {r[7]:.2f})" if r[7] is not None else ""
                                findings = r[6] if r[6] else "No findings recorded"
                                img_type = r[5].upper() if r[5] else "Unknown"
                                date_str = r[3].strftime('%Y-%m-%d')
                                st.markdown(f"""
**Date:** {date_str}
- **Type:** {img_type} ({r[4]} priority)
- **Findings:** {findings} {conf_str}
                                """)
                                st.divider()
                                studies_text += f"- Date: {date_str}, Type: {img_type}, Priority: {r[4]}, Findings: {findings}\n"
                        
                        st.markdown("---")
                        if st.button("📄 Generate Radiology Report"):
                            with st.spinner("Generating AI Radiology Report..."):
                                report_prompt = f"""
You are an expert AI radiologist. Based on the following patient details and recent study findings, generate a formal, professional radiology report.

Patient Name: {patient_info[1]}
Patient ID: {patient_info[0]}
Gender: {patient_info[2]}

Recent Studies:
{studies_text}

The report should include the following sections:
- Patient Information
- Clinical Indication
- Imaging Modalities
- Findings
- Impression

Do not output anything else but the report itself.
"""
                                report_llm = ChatOllama(model="tinyllama", temperature=0.1)
                                try:
                                    generated_report = report_llm.invoke(report_prompt).content
                                    st.session_state[f"report_{search_name}"] = generated_report
                                except Exception as e:
                                    st.error(f"Failed to generate report: {e}")
                        
                        if f"report_{search_name}" in st.session_state:
                            report_text = st.session_state[f"report_{search_name}"]
                            st.markdown("### 📝 AI Radiology Report")
                            st.text_area("Report Preview", report_text, height=300)
                            st.download_button(
                                label="⬇️ Download Report",
                                data=report_text,
                                file_name=f"Radiology_Report_{patient_info[1].replace(' ', '_')}.txt",
                                mime="text/plain"
                            )
                    else:
                        st.info("No studies found for this patient.")
                else:
                    st.warning("Patient not found in database.")
            except Exception as e:
                st.error(f"Error fetching patient data: {e}")


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
        return FAISS.load_local("./faiss_db", embeddings, allow_dangerous_deserialization=True)
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
                
                cur.execute("SELECT COUNT(*) FROM oads.patients")
                total_patients = cur.fetchone()[0]
                
                cur.execute("SELECT COUNT(*) FROM oads.studies")
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