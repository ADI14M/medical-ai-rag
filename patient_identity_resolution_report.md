# Medical AI RAG System: Patient-Identity Bug Resolution Report

This report documents the root cause, implementation details, and verification results for the patient-identity bug.

---

## 1. Root Cause Analysis

### The Bug
When a user queried the system with a prompt like *"Summarize patient Ananya Nair"*, the system would occasionally retrieve and summarize chunks belonging to other patients (such as Vikram Singh, Priya Reddy, or Sneha Iyer).

### Technical Cause
1. **Lack of Patient Constraint in Retrieval**: The RAG chatbot previously ran a standard vector search on the user's prompt over the entire FAISS database. It had no logic to parse patient names or apply metadata filtering.
2. **Missing Data Vector Scapegoat**: The database only contains image and analysis data for four patients (IDs: 1, 2, 3, and 3026). Other patients (such as Ananya Nair) have demographic records but no scan reports or analysis findings.
3. **Similarity Overlap**: Because Ananya Nair has no documents in the vector store, similarity search matched documents from the four active patients. Since there were no metadata constraints, these mismatched chunks were retrieved, populated into the prompt context, and summarized by the LLM as if they belonged to Ananya Nair.

---

## 2. Resolution Implementation Details

We resolved this bug by implementing a secure **Patient-Identity Verification & Retrieval Filtering pipeline**:

1. **Patient Extraction**:
   A regex-based parser scans the user query for mentions of patient IDs (e.g. `Patient 5`) and base names (e.g. `Ananya Nair`). If both are present, they are combined; if only one is present, it is extracted as a candidate.
2. **Exact Database Verification**:
   The candidate is looked up in the PostgreSQL `oads.patients` database using an exact query: `WHERE full_name = %s`.
   - If the patient does not exist, the query is blocked, and we return: `"No patient found with that name."`
   - If the patient exists, we retrieve the exact integer `patient_id`.
3. **Restricted SQL Retrieval (Hybrid Path)**:
   If a verified `patient_id` is present, the aggregate scans query in `app.py` is appended with: `AND p.patient_id = %s`.
4. **Restricted Vector Search (Standard Path)**:
   We enforce FAISS metadata filtering during search: `filter={"patient_id": verified_patient_id}`.
5. **Post-Retrieval Identity Verification**:
   Before building the LLM context, we double-check the metadata of all retrieved chunks. Any chunk that does not belong to the target `patient_id` is discarded automatically.
6. **Detailed Verification Logging**:
   We print verification logs (detected name, verified ID, retrieved chunk count, unique patient IDs found in retrieved chunks, and discarded chunk count) to the console and to the Streamlit UI's expander window (`st.expander("🛠️ RAG Debug Logs")`).

---

## 3. Code Modifications

### A. Helper Function `extract_and_verify_patient` (Implemented in `app.py` & `evaluate_model.py`)
```python
def extract_and_verify_patient(prompt):
    from config import DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT
    query_lower = prompt.lower()
    
    # 1. Check if patient query
    patient_keywords = ["patient", "summarize", "history", "insights", "report", "findings for", "results for", "detail for"]
    is_patient_query = any(w in query_lower for w in patient_keywords)
    
    base_names = ["priya reddy", "rahul verma", "sneha iyer", "karthik rao", "ananya nair", "vikram singh", "meera joshi", "rohan kulkarni", "divya menon", "arjun sharma"]
    mentioned_base_name = None
    for bn in base_names:
        if bn in query_lower:
            mentioned_base_name = bn.title()
            break
            
    match_id = re.search(r'\bpatient\s+(?:id\s+)?(\d+)\b', query_lower)
    if not match_id:
        match_id = re.search(r'\b(?:patient\s+)?(?:id\s*)?#?(\d+)\b', query_lower)
        
    patient_id_num = None
    if match_id:
        patient_id_num = int(match_id.group(1))

    if not mentioned_base_name and not patient_id_num:
        return False, None, None, "not_patient_query", None

    detected_name = None
    if patient_id_num and mentioned_base_name:
        detected_name = f"Patient {patient_id_num} - {mentioned_base_name}"
    elif patient_id_num:
        detected_name = f"Patient {patient_id_num}"
    elif mentioned_base_name:
        detected_name = mentioned_base_name

    # Handle full name pattern
    match_full = re.search(r'\bpatient\s+(\d+)\s*-\s*([a-z\s]+)\b', query_lower)
    if match_full:
        full_id = int(match_full.group(1))
        full_base = match_full.group(2).strip().title()
        detected_name = f"Patient {full_id} - {full_base}"
        patient_id_num = full_id
        mentioned_base_name = full_base

    try:
        conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD, port=DB_PORT)
        cur = conn.cursor()
        
        if patient_id_num and mentioned_base_name:
            exact_full_name = f"Patient {patient_id_num} - {mentioned_base_name}"
            cur.execute("SELECT patient_id, full_name FROM oads.patients WHERE full_name = %s", (exact_full_name,))
            row = cur.fetchone()
            if row:
                conn.close()
                return True, row[0], row[1], "verified", exact_full_name
            else:
                conn.close()
                return True, None, None, "not_found", exact_full_name

        if patient_id_num and not mentioned_base_name:
            cur.execute("SELECT patient_id, full_name FROM oads.patients WHERE patient_id = %s", (patient_id_num,))
            row = cur.fetchone()
            if row:
                conn.close()
                return True, row[0], row[1], "verified", f"Patient {patient_id_num}"
            else:
                conn.close()
                return True, None, None, "not_found", f"Patient {patient_id_num}"

        if mentioned_base_name and not patient_id_num:
            # Task: Perform exact database lookup using full_name
            cur.execute("SELECT patient_id, full_name FROM oads.patients WHERE full_name = %s", (mentioned_base_name,))
            row = cur.fetchone()
            if row:
                conn.close()
                return True, row[0], row[1], "verified", mentioned_base_name
            else:
                conn.close()
                return True, None, None, "not_found", mentioned_base_name
                
        conn.close()
    except Exception as e:
        print(f"Error checking patient identity: {e}")
        
    return True, None, None, "not_found", detected_name
```

### B. Standard Path Vector Search Constraint (Enforced in `app.py` & `evaluate_model.py`)
#### Before:
```python
search_kwargs = {"k": TOP_K, "fetch_k": FETCH_K, "lambda_mult": 0.5}
retriever = st.session_state.vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs=search_kwargs
)
docs = retriever.invoke(prompt)
```
#### After:
```python
search_kwargs = {"k": TOP_K, "fetch_k": FETCH_K, "lambda_mult": 0.5}
if verified_patient_id is not None:
    search_kwargs["filter"] = {"patient_id": verified_patient_id}

retriever = st.session_state.vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs=search_kwargs
)
docs = retriever.invoke(prompt)
```

### C. Post-Retrieval Verification & Discarding (Enforced in `app.py` & `evaluate_model.py`)
```python
# Before sending context to the LLM: Verify and discard chunks belonging to other patients
for doc in docs:
    chunk_pid = doc.metadata.get("patient_id")
    if verified_patient_id is not None and chunk_pid != verified_patient_id:
        discarded_count += 1
        if chunk_pid is not None:
            mismatched_pids.add(chunk_pid)
        continue
        
    if len(context) + len(doc.page_content) < MAX_CONTEXT_CHARS:
        context += doc.page_content + "\n\n"
        used_docs.append(doc)
    else:
        break
```

---

## 4. Verification Results

We verified the resolution logic on multiple test query scenarios:

| User Query | Detected Patient | Lookup Status | Verified Patient ID | Action / LLM Output |
| :--- | :--- | :--- | :--- | :--- |
| **"Summarize patient Ananya Nair"** | `Ananya Nair` | `not_found` (mismatch to DB `Patient X - Ananya Nair`) | `None` | Blocked & returned: `"No patient found with that name."` |
| **"Summarize patient Patient 5 - Ananya Nair"** | `Patient 5 - Ananya Nair` | `verified` | `5` | Queries FAISS with `patient_id=5` filter. Returns 0 chunks. Safely outputs `"Not found in database"`. |
| **"Summarize patient Patient 3 - Sneha Iyer"** | `Patient 3 - Sneha Iyer` | `verified` | `3` | Queries FAISS with `patient_id=3` filter. Retrieves and summarizes Patient 3's records. |
| **"Summarize patient Patient 3"** | `Patient 3` | `verified` (mapped to Sneha Iyer) | `3` | Mapped and verified. Retrieves and summarizes Patient 3's records. |
| **"What are the common symptoms of pneumonia?"** | `None` | `not_patient_query` | `None` | Processed as a general medical query (no filtering). |
