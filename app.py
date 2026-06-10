import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import time
import re
import psycopg2

def extract_and_verify_patient(prompt):
    """
    Parses the prompt to detect any patient name or ID.
    If a patient name or ID is present, performs dynamic lookup & exact database verification.
    Returns:
        is_patient_query (bool): True if the query targets a patient.
        patient_id (int or None): The verified patient ID.
        patient_name (str or None): The verified patient name.
        lookup_status (str): 'verified' or 'not_found' or 'not_patient_query'.
        detected_name (str or None): The exact name string that was resolved.
    """
    import re
    import psycopg2
    from config import DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT

    query_lower = prompt.lower()
    
    # 1. Check for ID (e.g. "Patient 3" or just "3")
    match_id = re.search(r'\bpatient\s+(?:id\s+)?(\d+)\b', query_lower)
    if not match_id:
        match_id = re.search(r'\b(?:patient\s+)?(?:id\s*)?#?(\d+)\b', query_lower)
        
    patient_id_num = None
    if match_id:
        patient_id_num = int(match_id.group(1))

    # 2. Stopword filtering to extract name candidate words
    words = re.findall(r'\b[a-zA-Z]+\b', prompt)
    stopwords = {
        # Query intent words
        "patient", "patients", "summarize", "history", "insights", "report",
        "findings", "results", "detail", "details", "visit", "visits", "multiple",
        "retrieve", "dates", "recorded", "identify", "detect", "generate",
        "structured", "draft", "formal", "impression", "impressions", "recommendations",
        "recommendation", "suggest", "steps", "provide", "list", "show", "give", "get",
        "me", "us",
        # Common prepositions / articles / verbs
        "of", "the", "a", "an", "is", "are", "what", "recent", "who", "has",
        "having", "which", "had", "for", "with", "on", "based", "their", "all",
        "there", "any", "does", "have", "made", "next", "do", "in", "and", "or",
        # Clinical / imaging terms (modalities, adjectives)
        "scan", "scans", "scanning", "abnormal", "imaging", "priority", "levels",
        "studies", "study", "result", "flagged", "abnormality", "minor", "issue",
        "issues", "clinical", "ct", "mri", "xray", "mr", "ultrasound", "x",
        "radiology", "radiological"
    }
    candidate_words = [w for w in words if w.lower() not in stopwords]
    candidate_words.sort(key=len, reverse=True)

    # If neither candidate name words nor ID are found, it's not a patient-specific query
    if not candidate_words and not patient_id_num:
        return False, None, None, "not_patient_query", None

    candidate_name = None

    try:
        conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD, port=DB_PORT)
        cur = conn.cursor()

        # A. Resolve using patient ID if explicitly specified
        if patient_id_num:
            cur.execute("SELECT full_name FROM oads.patients WHERE patient_id = %s", (patient_id_num,))
            row = cur.fetchone()
            if row:
                candidate_name = row[0]

        # B. If not resolved yet, resolve using name phrases dynamically (e.g. "Priya Reddy")
        if not candidate_name and len(candidate_words) >= 2:
            for i in range(len(candidate_words) - 1):
                phrase = f"{candidate_words[i]} {candidate_words[i+1]}"
                cur.execute("SELECT full_name FROM oads.patients WHERE full_name ILIKE %s ORDER BY patient_id LIMIT 1", (f"%{phrase}%",))
                row = cur.fetchone()
                if row:
                    candidate_name = row[0]
                    break

        # C. If not resolved yet, resolve using single names (e.g. "rahul")
        if not candidate_name:
            for word in candidate_words:
                if len(word) < 3:
                    continue
                cur.execute("SELECT full_name FROM oads.patients WHERE full_name ILIKE %s ORDER BY patient_id LIMIT 1", (f"%{word}%",))
                row = cur.fetchone()
                if row:
                    candidate_name = row[0]
                    break

        if not candidate_name:
            conn.close()
            # If a patient name was mentioned but could not be mapped to any record, return not_found
            return True, None, None, "not_found", " ".join(candidate_words) if candidate_words else f"Patient ID {patient_id_num}"

        # D. Execute exact verification query:
        # SELECT patient_id, full_name FROM oads.patients WHERE LOWER(full_name)=LOWER(%s)
        cur.execute("""
            SELECT patient_id, full_name
            FROM oads.patients
            WHERE LOWER(full_name) = LOWER(%s)
        """, (candidate_name,))
        row = cur.fetchone()
        conn.close()

        if row:
            return True, row[0], row[1], "verified", candidate_name
        else:
            return True, None, None, "not_found", candidate_name

    except Exception as e:
        print(f"Error checking patient identity: {e}")
        return True, None, None, "not_found", " ".join(candidate_words) if candidate_words else None

def parse_chunk_text(text, metadata=None):
    pid = None
    study_date = ""
    image_type = ""
    findings = ""
    priority = ""
    confidence = ""
    
    # Parse lines
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("Patient ID:"):
            try:
                pid = int(line.split(":", 1)[1].strip())
            except:
                pass
        elif line.startswith("Study Date:"):
            study_date = line.split(":", 1)[1].strip()
        elif line.startswith("Image Type:"):
            image_type = line.split(":", 1)[1].strip()
        elif line.startswith("Findings:"):
            findings = line.split(":", 1)[1].strip()
        elif line.startswith("Study Priority:"):
            priority = line.split(":", 1)[1].strip()
        elif line.startswith("AI Confidence Score:"):
            confidence = line.split(":", 1)[1].strip()
            
    # Fallback to metadata
    if pid is None and metadata and "patient_id" in metadata:
        pid = int(metadata["patient_id"])
    if not study_date and metadata and "study_date" in metadata:
        study_date = str(metadata["study_date"]).strip()
    if not image_type and metadata and "image_type" in metadata:
        image_type = str(metadata["image_type"]).strip()
    if not priority and metadata and "priority" in metadata:
        priority = str(metadata["priority"]).strip()
        
    return pid, study_date, image_type, findings, priority, confidence

def aggregate_chunks_into_summary(patient_name, patient_id, gender, unique_parsed_chunks):
    summary_parts = [
        f"Patient ID: {patient_id}",
        f"Patient Name: {patient_name}",
        f"Gender: {gender}",
        "Imaging Studies & Clinical History:"
    ]
    for chunk in unique_parsed_chunks:
        pid, study_date, image_type, findings, priority, confidence = chunk
        confidence_str = f" (Confidence: {confidence})" if confidence else ""
        priority_str = f" ({priority} priority)" if priority else ""
        summary_parts.append(f"- Date: {study_date} | Type: {image_type.upper()}{priority_str} | Findings: {findings}{confidence_str}")
        
    return "\n".join(summary_parts)

def clean_llm_output(text):
    """
    Cleans the LLM output to prevent exposing prompt headers and structures.
    Only the final clinical answer / summary is shown to the user.
    """
    marker = "final clinical answer:"
    lower_text = text.lower()
    if marker in lower_text:
        idx = lower_text.rfind(marker)
        return text[idx + len(marker):].strip()
        
    # Fallback: strip standard prompt sections from output
    headers = [
        "[ DATABASE STATISTICS ]",
        "[ PATIENT IMAGING MAPPING ]",
        "[ PATIENT LAB EVENT CONTEXT ]",
        "[ STRICT RULES ]"
    ]
    lines = text.split("\n")
    cleaned_lines = []
    skip_mode = False
    for line in lines:
        stripped = line.strip()
        if any(h in stripped for h in headers):
            skip_mode = True
            continue
        if stripped.startswith("Question:") or stripped.startswith("Final Clinical Answer:"):
            skip_mode = False
            continue
        if skip_mode:
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()




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
                SELECT p.patient_id, p.full_name, p.gender,
                       s.study_date, s.priority,
                       COALESCE(i.image_type, 'Unknown') AS image_type,
                       COALESCE(a.findings_summary, 'No clinical findings recorded') AS findings_summary,
                       a.confidence_score
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
                        if st.button("📄 Generate Radiology Report (PDF)"):
                            with st.spinner("Generating AI Radiology Report..."):
                                # Collect study rows for PDF: (study_date, priority, image_type, findings, confidence)
                                pdf_study_rows = []
                                for r in rows:
                                    if r[3]:  # has study_date
                                        pdf_study_rows.append((
                                            r[3],                                     # study_date
                                            r[4],                                     # priority
                                            r[5] if r[5] else "Unknown",              # image_type
                                            r[6] if r[6] else "No findings recorded", # findings
                                            r[7],                                     # confidence
                                        ))

                                # Generate LLM impression & recommendations
                                impression_text = ""
                                recommendations_text = ""
                                try:
                                    # Always use clinical model for radiology report generation
                                    report_llm = ChatOllama(model="medgemma1.5:4b", temperature=0.1)

                                    impression_prompt = f"""Based on these radiology findings for patient {patient_info[1]}, write a brief clinical impression in 2-3 sentences. Only output the impression, nothing else.

Findings:
{studies_text}"""
                                    impression_text = report_llm.invoke(impression_prompt).content.strip()

                                    recs_prompt = f"""Based on these radiology findings for patient {patient_info[1]}, write 2-3 specific clinical recommendations. Only output the recommendations, nothing else.

Findings:
{studies_text}"""
                                    recommendations_text = report_llm.invoke(recs_prompt).content.strip()
                                except Exception:
                                    pass  # fallback to auto-generated text in the PDF

                                from generate_report import generate_pdf_report

                                pdf_bytes = generate_pdf_report(
                                    patient_id=patient_info[0],
                                    patient_name=patient_info[1],
                                    gender=patient_info[2] or "Unknown",
                                    study_rows=pdf_study_rows,
                                    ai_impression=impression_text,
                                    ai_recommendations=recommendations_text,
                                )
                                st.session_state[f"pdf_report_{search_name}"] = pdf_bytes
                                st.session_state[f"pdf_report_name_{search_name}"] = patient_info[1]
                                st.success("✅ PDF report generated successfully!")

                        if f"pdf_report_{search_name}" in st.session_state:
                            pdf_data = st.session_state[f"pdf_report_{search_name}"]
                            report_patient_name = st.session_state.get(f"pdf_report_name_{search_name}", "Patient")
                            safe_name = report_patient_name.replace(" ", "_")

                            st.download_button(
                                label="⬇️ Download Radiology Report (PDF)",
                                data=pdf_data,
                                file_name=f"Radiology_Report_{safe_name}.pdf",
                                mime="application/pdf",
                            )
                    else:
                        st.info("No studies found for this patient.")
                else:
                    st.warning("Patient not found in database.")
            except Exception as e:
                st.error(f"Error fetching patient data: {e}")


# ====================== Settings ======================
TEMPERATURE = 0.2

def get_llm(question):
    """Routes the question to the appropriate model based on clinical intent."""
    q_lower = question.lower()
    clinical_keywords = [
        "summarize", "radiology", "ct", "mri", "abnormalities", "clinical", 
        "findings", "diagnosis", "recommendations", "scan", "xray", "ultrasound",
        "impression", "report", "patient", "history", "health", "disease", "treatment"
    ]
    if any(k in q_lower for k in clinical_keywords):
        return "medgemma1.5:4b"
    return "phi3"


TOP_K = 10
FETCH_K = 40
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

        # Initialize debug logging variables
        debug_sql_query = "N/A"
        debug_sql_records = 0
        debug_faiss_chunks = 0

        # ====================== PATIENT DETECTION & VERIFICATION ======================
        is_patient_query, verified_patient_id, verified_patient_name, lookup_status, detected_name = extract_and_verify_patient(prompt)

        # Safety Check / Validation check: If patient is requested but does not exist in DB
        # Requirement 4: If no patient exists, Return: "Patient not found."
        if is_patient_query and lookup_status == "not_found":
            full_response = "Patient not found."
            placeholder.markdown(full_response)
            
            # Print minimal logs for validation failure
            print("\n=== RAG PIPELINE DEBUG LOGS ===")
            print(f"Detected patient name: {detected_name}")
            print(f"SQL query executed: SELECT patient_id, full_name FROM oads.patients WHERE LOWER(full_name)=LOWER(%s)")
            print(f"Retrieved patient_id: None")
            print(f"Retrieved chunk count: 0")
            print(f"Unique patient_ids found: []")
            print("===============================\n")
            
            # Render in Streamlit UI
            with st.expander("Developer Debug Logs", expanded=False):
                st.markdown(f"**Detected Patient Name:** `{detected_name}`")
                st.markdown(f"**Verified Patient ID:** `None`")
                st.markdown(f"**SQL Query Executed:** `SELECT patient_id, full_name FROM oads.patients WHERE LOWER(full_name)=LOWER(%s)`")
                st.markdown(f"**Status:** `Validation Failed - Patient Not Found`")
        else:
            # ====================== RETRIEVAL & COMBINATION ======================
            context = ""
            combined_chunks = []
            
            # Keep track of unique pids for logs
            retrieved_pids = set()
            
            # If a patient is found, restrict all retrieval to that patient_id (SQL first, FAISS second)
            if verified_patient_id is not None:
                # 1. SQL Filtering First
                sql_query = """
                    SELECT
                        p.patient_id, p.full_name, p.gender,
                        s.study_date, s.priority,
                        COALESCE(i.image_type, 'Unknown') AS image_type,
                        COALESCE(a.findings_summary, 'No clinical findings recorded') AS findings_summary,
                        a.confidence_score
                    FROM oads.patients p
                    JOIN oads.studies s ON p.patient_id = s.patient_id
                    JOIN oads.images i ON s.study_id = i.study_id
                    LEFT JOIN oads.analysis a ON i.image_id = a.image_id
                    WHERE p.patient_id = %s
                    ORDER BY s.study_date DESC
                """
                debug_sql_query = sql_query.strip()

                patient_gender = "Unknown"
                try:
                    import psycopg2
                    from config import DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT
                    conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD, port=DB_PORT)
                    cur = conn.cursor()
                    cur.execute(sql_query, (verified_patient_id,))
                    sql_rows = cur.fetchall()
                    conn.close()

                    debug_sql_records = len(sql_rows)

                    if sql_rows:
                        patient_gender = sql_rows[0][2] or "Unknown"
                    else:
                        # Fetch gender if no studies exist
                        conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD, port=DB_PORT)
                        cur = conn.cursor()
                        cur.execute("SELECT gender FROM oads.patients WHERE patient_id = %s", (verified_patient_id,))
                        row = cur.fetchone()
                        conn.close()
                        if row:
                            patient_gender = row[0] or "Unknown"

                    # ── STEP 4/5 guard: if no studies/images exist, report cleanly ──
                    if not sql_rows:
                        full_response = f"No clinical records found for patient {verified_patient_name} (ID: {verified_patient_id})."
                        placeholder.markdown(full_response)
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                        st.stop()

                    # Check if ALL findings are the 'no findings' placeholder (shouldn't happen
                    # after the integrity fix, but guards against future partial runs)
                    real_findings = [r[6] for r in sql_rows if r[6] and r[6] != 'No clinical findings recorded']
                    if not real_findings:
                        full_response = f"No clinical findings available for patient {verified_patient_name}."
                        placeholder.markdown(full_response)
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                        st.stop()

                    # Format SQL rows as chunks
                    for r in sql_rows:
                        pid, fname, gender, study_date, priority, img_type, findings, confidence = r
                        if hasattr(study_date, "strftime"):
                            study_date = study_date.strftime("%Y-%m-%d")
                        confidence_str = f"{confidence:.2f}" if confidence is not None else "N/A"
                        findings_str = findings if findings else "No clinical findings recorded"

                        chunk_text = (
                            f"Patient ID: {pid}\nPatient Name: {fname}\nGender: {gender}\n"
                            f"Study Date: {study_date}\nStudy Priority: {priority}\n"
                            f"Image Type: {img_type}\nFindings: {findings_str}\n"
                            f"AI Confidence Score: {confidence_str}\n"
                        )
                        meta = {
                            "patient_id": pid,
                            "study_date": study_date,
                            "image_type": img_type,
                            "findings": findings_str,
                            "priority": priority,
                            "confidence": confidence_str
                        }
                        combined_chunks.append((chunk_text, meta))
                        retrieved_pids.add(pid)
                except Exception as e:
                    debug_sql_query = f"Error in SQL query: {e}"
                    debug_sql_records = 0
                    
                # 2. FAISS Retrieval Second
                try:
                    search_kwargs = {"k": TOP_K, "fetch_k": FETCH_K, "lambda_mult": 0.5, "filter": {"patient_id": verified_patient_id}}
                    retriever = st.session_state.vectorstore.as_retriever(
                        search_type="mmr",
                        search_kwargs=search_kwargs
                    )
                    docs = retriever.invoke(prompt)
                    debug_faiss_chunks = len(docs)
                    for doc in docs:
                        combined_chunks.append((doc.page_content, doc.metadata))
                        if "patient_id" in doc.metadata:
                            retrieved_pids.add(doc.metadata["patient_id"])
                except Exception as e:
                    debug_faiss_chunks = 0
            
            else:
                # General query or aggregate modality query
                # Detect if aggregate modality query
                query_lower = prompt.lower()
                matched_modality = None
                for m in ["ct", "mri", "xray", "x-ray", "ultrasound"]:
                    if m in query_lower:
                        matched_modality = m
                        break
                        
                is_list_query = any(w in query_lower for w in ["patients with", "who has", "patients having", "list of patients", "which patients", "who had"])
                
                if is_list_query and matched_modality:
                    # Hybrid Path: Query database directly for all patients with this scan type
                    sql_query = """
                        SELECT p.patient_id, p.full_name, string_agg(DISTINCT i.image_type, ', ') as scan_types, string_agg(DISTINCT a.findings_summary, '; ') as findings
                        FROM oads.patients p
                        JOIN oads.studies s ON p.patient_id = s.patient_id
                        JOIN oads.images i ON s.study_id = i.study_id
                        LEFT JOIN oads.analysis a ON i.image_id = a.image_id
                        WHERE i.image_type ILIKE %s
                        GROUP BY p.patient_id, p.full_name
                        ORDER BY p.patient_id;
                    """
                    debug_sql_query = sql_query.strip()
                    
                    try:
                        import psycopg2
                        import math
                        from config import DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT
                        conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD, port=DB_PORT)
                        cur = conn.cursor()
                        sql_modality = "xray" if matched_modality in ["xray", "x-ray"] else matched_modality
                        cur.execute(sql_query, (sql_modality,))
                        rows = cur.fetchall()
                        conn.close()
                        
                        debug_sql_records = len(rows)
                        debug_faiss_chunks = 0
                        
                        # Text-based pagination details
                        total_count = len(rows)
                        page_size = 10
                        total_pages = math.ceil(total_count / page_size) if total_count > 0 else 1
                        page = 1  # Default to page 1
                        
                        context_parts = [
                            f"Total Patients with {matched_modality.upper()} scans found in database: {total_count}.",
                            f"Showing Page {page} of {total_pages}:"
                        ]
                        
                        start_idx = (page - 1) * page_size
                        end_idx = start_idx + page_size
                        page_rows = rows[start_idx:end_idx]
                        
                        for idx, r in enumerate(page_rows):
                            pid, name, scan_types, findings = r
                            findings_str = findings if findings else "No findings recorded"
                            context_parts.append(f"{start_idx + idx + 1}. Patient ID: {pid} | Name: {name} | Modalities: {scan_types.upper()} | Findings: {findings_str}")
                            retrieved_pids.add(pid)
                            
                        context = "\n".join(context_parts)
                    except Exception as e:
                        context = "Error retrieving matching patients from database."
                        debug_sql_query = f"Error in SQL query: {e}"
                else:
                    # Standard Path: MMR Vector Search for general questions
                    search_kwargs = {"k": TOP_K, "fetch_k": FETCH_K, "lambda_mult": 0.5}
                    retriever = st.session_state.vectorstore.as_retriever(
                        search_type="mmr",
                        search_kwargs=search_kwargs
                    )
                    docs = retriever.invoke(prompt)
                    
                    debug_faiss_chunks = len(docs)
                    debug_sql_query = "N/A (Standard vector search executed)"
                    debug_sql_records = 0
                    
                    for doc in docs:
                        combined_chunks.append((doc.page_content, doc.metadata))
                        if "patient_id" in doc.metadata:
                            retrieved_pids.add(doc.metadata["patient_id"])

            # ====================== VERIFICATION & DEDUPLICATION ======================
            verified_chunks = []
            seen_keys = set()
            discarded_count = 0
            mismatched_pids = set()
            
            # Before sending context to the LLM: Verify and deduplicate chunks
            for text, meta in combined_chunks:
                pid, study_date, image_type, findings, priority, confidence = parse_chunk_text(text, meta)
                
                # 5. Verify that all retrieved chunks belong to the same verified_patient_id (if patient query)
                if verified_patient_id is not None and pid != verified_patient_id:
                    discarded_count += 1
                    if pid is not None:
                        mismatched_pids.add(pid)
                    continue
                
                # 6. Deduplicate by patient_id + study_date + image_type + findings
                key = (pid, str(study_date).strip().lower(), str(image_type).strip().lower(), str(findings).strip().lower())
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                
                verified_chunks.append((pid, study_date, image_type, findings, priority, confidence))
                
            # ====================== AGGREGATION & CONTEXT BUILDING ======================
            if verified_patient_id is not None:
                # 7. If multiple chunks belong to the same patient, aggregate them into a single summary
                if verified_chunks:
                    context = aggregate_chunks_into_summary(verified_patient_name, verified_patient_id, patient_gender, verified_chunks)
                else:
                    context = "Not found in database"
            elif not (is_list_query and matched_modality):
                # Standard path context formatting
                context_parts = []
                for chunk in verified_chunks:
                    pid, study_date, image_type, findings, priority, confidence = chunk
                    confidence_str = f"Score: {confidence}" if confidence else ""
                    context_parts.append(f"Patient ID: {pid}\nStudy Date: {study_date}\nImage Type: {image_type}\nFindings: {findings}\n{confidence_str}")
                context = "\n\n".join(context_parts)
                
            # ====================== DB STATS ======================
            if "db_stats" not in st.session_state:
                try:
                    import psycopg2
                    from config import DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT
                    
                    # Connect to ehr_db for live EHR counts
                    conn_ehr = psycopg2.connect(host=DB_HOST, database='ehr_db', user=DB_USER, password=DB_PASSWORD, port=DB_PORT)
                    cur_ehr = conn_ehr.cursor()
                    cur_ehr.execute("SELECT COUNT(*) FROM ehr.patients")
                    total_patients = cur_ehr.fetchone()[0]
                    cur_ehr.execute("SELECT COUNT(*) FROM ehr.labevents")
                    total_labevents = cur_ehr.fetchone()[0]
                    cur_ehr.close()
                    conn_ehr.close()
                    
                    # Connect to oads_db for imaging mappings
                    conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD, port=DB_PORT)
                    cur = conn.cursor()
                    cur.execute("""
                        SELECT DISTINCT p.patient_id, p.full_name, i.image_type
                        FROM oads.patients p
                        JOIN oads.studies s ON p.patient_id = s.patient_id
                        JOIN oads.images i ON s.study_id = i.study_id
                        ORDER BY p.patient_id
                    """)
                    rows = cur.fetchall()
                    conn.close()
                    
                    patient_map = {}
                    for pid, name, img_type in rows:
                        if name not in patient_map:
                            patient_map[name] = []
                        if img_type:
                            patient_map[name].append(img_type.upper())
                    
                    summary_lines = ["[ PATIENT IMAGING MAPPING ]"]
                    for name, modalities in patient_map.items():
                        summary_lines.append(f"- {name}: {', '.join(modalities)}")
                    imaging_summary = "\n".join(summary_lines)
                    
                    st.session_state.db_stats = {
                        "total_patients": total_patients,
                        "total_labevents": total_labevents,
                        "imaging_summary": imaging_summary
                    }
                except Exception as e:
                    print(f"Error fetching DB statistics: {e}")
                    st.session_state.db_stats = {
                        "total_patients": "Unknown",
                        "total_labevents": "Unknown",
                        "imaging_summary": ""
                    }

            stats = st.session_state.db_stats

            # ====================== PROMPT ======================
            prompt_template = ChatPromptTemplate.from_template("""
You are a comprehensive, highly detailed medical assistant.

[ DATABASE STATISTICS ]
Total Registered Patients: {total_patients}
Total EHR Lab Events: {total_labevents}
{imaging_summary}

[ PATIENT LAB EVENT CONTEXT ]
{context}

[ STRICT RULES ]
1. Answer using ONLY natural language. NEVER output raw SQL queries or database code.
2. Provide a detailed and comprehensive clinical summarization of the retrieved context. Explicitly point out any values flagged as ABNORMAL and expand on their potential significance.
3. Provide cohesive, thoroughly explained patient insights based on the retrieved lab events and imaging.
4. Output a rich, structured analytical answer. Explain your findings clearly to the user in a professional medical tone.
5. If the answer cannot be confidently deduced from the Context or Database Statistics, output exactly: "Not found in database".

Question: {question}

Final Clinical Answer:
""")

            final_prompt = prompt_template.format(
                total_patients=stats["total_patients"],
                total_labevents=stats["total_labevents"],
                imaging_summary=stats.get("imaging_summary", ""),
                context=context,
                question=prompt
            )

            # ====================== LLM ======================
            selected_model = get_llm(prompt)
            llm = ChatOllama(
                model=selected_model,
                temperature=TEMPERATURE,
                num_ctx=2048,  # Hard cap memory allocation to stay under 2.5 GiB
                repeat_penalty=1.2, # Strictly penalize repetitive phrases to prevent loops
                top_k=40,
                top_p=0.9
            )

            stream = llm.stream(final_prompt)

            for chunk in stream:
                if hasattr(chunk, "content"):
                    full_response += chunk.content
                else:
                    full_response += str(chunk)

                display_response = clean_llm_output(full_response)
                if display_response:
                    placeholder.markdown(display_response + "▌")
                else:
                    placeholder.markdown("*Thinking...*")

            final_cleaned_response = clean_llm_output(full_response)
            placeholder.markdown(final_cleaned_response)

            # ====================== TIME ======================
            end_time = time.time()
            st.caption(f"⏱️ {(end_time - start_time):.2f} sec")

            # ====================== DEBUG LOGGING ======================
            # Print to terminal console
            debug_detected_name = detected_name if detected_name else "None"
            debug_verified_id = verified_patient_id if verified_patient_id else "None"
            
            # Retrieved chunk count is sum of sql rows and faiss chunks retrieved
            retrieved_chunk_count = debug_sql_records + debug_faiss_chunks
            
            print("\n=== RAG PIPELINE DEBUG LOGS ===")
            print(f"Detected patient name: {debug_detected_name}")
            print(f"SQL query executed: {debug_sql_query}")
            print(f"Retrieved patient_id: {debug_verified_id}")
            print(f"Retrieved chunk count: {retrieved_chunk_count}")
            print(f"Unique patient_ids found: {list(retrieved_pids)}")
            print(f"Selected Model: {selected_model}")
            print("===============================\n")



    st.session_state.messages.append({
        "role": "assistant",
        "content": final_cleaned_response if verified_patient_id is not None or not is_patient_query or lookup_status != "not_found" else "Patient not found."
    })