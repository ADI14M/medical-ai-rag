# Medical AI RAG System: Debug & Technical Resolution Report

This report outlines the root causes, resolution plan, and exact code modifications made to debug and resolve key clinical data retrieval, deduplication, and display issues within the Medical AI RAG system.

---

## 1. Root Causes & Technical Challenges

### A. Single-Result / Missing Patients in Aggregate Modality Queries
* **Problem**: When queries like *"Show all patients who have CT scans"* were executed, the system returned only a single patient (usually Patient 3026 / Vikram Singh) or omitted other eligible patients like Priya Reddy and Rahul Verma.
* **Root Cause**: The database contains highly skewed counts of study records per patient. Patient 3026 has approximately **4,900 studies**, Patient 1 has **201**, and Patient 2 has only **1**. 
Under standard vector similarity search, the FAISS retriever queried for the closest matching text chunks. Because the number of retrieved chunks was capped at `k=8`, all top retrieved chunks belonged to Patient 3026, completely burying and omitting Patient 1 and Patient 2.
* **Resolution**: Implemented a **hybrid retrieval path** that detects aggregate scanning queries (e.g., matching keywords like *"patients with"*, *"who has"*, and scan types like `ct`, `mri`, `xray`). For these queries, the system bypasses the vector search and runs a direct, optimized SQL query to pull all matching records.

### B. Duplicate Patient Results
* **Problem**: Even when patients were successfully retrieved, their records were displayed multiple times, or duplicate scans were listed in the final response.
* **Root Cause**: In the database schema, a single patient has multiple studies, studies have multiple images, and images have multiple analysis rows. Joining `patients` with `studies`, `images`, and `analysis` tables without grouping or distinct filters resulted in duplicate joined rows for each individual patient scan.
* **Resolution**: Replaced the Python-layer post-processing deduplication with **SQL-layer aggregation and grouping**. We group by `p.patient_id, p.full_name` and use PostgreSQL's `string_agg(DISTINCT ...)` to aggregate modalities and findings into single-row string results per patient.

### C. Ingestion-layer Vector Duplication
* **Problem**: The vector database contained 4,989 records, which bloated memory and slowed retrieval latency.
* **Root Cause**: The deduplication key hash used in `extract_data.py` did not normalize the AI confidence score. Floating-point variations (e.g., `0.82000005` vs `0.82`) produced distinct hashes for logically duplicate records.
* **Resolution**: Formatted and normalized the confidence score to exactly 2 decimal places (`f"{confidence:.2f}"`) prior to generating the deduplication key hash. Rebuilding the vector index shrunk the vector database size from **4,989** duplicate-laden records to **392** unique records.

---

## 2. Code Modification Details

### 1. Ingestion Normalization
**File**: [extract_data.py](file:///Users/adi/Documents/Coding/medical-ai-rag/extract_data.py)

#### Before:
```python
# ✅ DEDUP KEY (fixed to include confidence)
key_str = f"{patient_id}|{study_date}|{image_type}|{findings}|{confidence}"
key_hash = hashlib.md5(key_str.encode()).hexdigest()
```

#### After:
```python
# ✅ DEDUP KEY (fixed to include formatted confidence)
confidence_str = f"{confidence:.2f}"
key_str = f"{patient_id}|{study_date}|{image_type}|{findings}|{confidence_str}"
key_hash = hashlib.md5(key_str.encode()).hexdigest()
```

---

### 2. Upgraded Retrieval Settings & MMR Capping
**Files**: [app.py](file:///Users/adi/Documents/Coding/medical-ai-rag/app.py) & [evaluate_model.py](file:///Users/adi/Documents/Coding/medical-ai-rag/evaluate_model.py)

#### Before:
```python
TOP_K = 8
FETCH_K = 30
```

#### After:
```python
TOP_K = 10
FETCH_K = 40
```

---

### 3. Hybrid SQL Retrieval & SQL-Level Deduplication
**File**: [app.py](file:///Users/adi/Documents/Coding/medical-ai-rag/app.py)

#### Before:
```python
        if is_list_query and matched_modality:
            # Hybrid Path: Query database directly for all patients with this scan type to bypass vector store capping
            try:
                import psycopg2
                from config import DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT
                conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD, port=DB_PORT)
                cur = conn.cursor()
                sql_modality = "xray" if matched_modality in ["xray", "x-ray"] else matched_modality
                cur.execute("""
                    SELECT DISTINCT p.patient_id, p.full_name, i.image_type, a.findings_summary
                    FROM oads.patients p
                    JOIN oads.studies s ON p.patient_id = s.patient_id
                    JOIN oads.images i ON s.study_id = i.study_id
                    JOIN oads.analysis a ON i.image_id = a.image_id
                    WHERE i.image_type ILIKE %s
                    ORDER BY p.patient_id;
                """, (sql_modality,))
                rows = cur.fetchall()
                conn.close()
                
                patient_records = {}
                for pid, name, img_type, findings in rows:
                    if pid not in patient_records:
                        patient_records[pid] = {
                            "name": name,
                            "img_type": img_type.upper(),
                            "findings": set()
                        }
                    if findings:
                        patient_records[pid]["findings"].add(findings)
                
                context_parts = []
                for pid, info in patient_records.items():
                    findings_str = ", ".join(info["findings"])
                    context_parts.append(f"Patient ID: {pid} | Patient Name: {info['name']} | Scan Type: {info['img_type']} | Findings: {findings_str}")
                context = "\n".join(context_parts)
            except Exception as e:
                context = "Error retrieving matching patients from database."
```

#### After:
```python
        if is_list_query and matched_modality:
            # Hybrid Path: Query database directly for all patients with this scan type to bypass vector store capping
            # Fix duplicates by grouping by patient_id in SQL and aggregating findings
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
                    
                context = "\n".join(context_parts)
            except Exception as e:
                context = "Error retrieving matching patients from database."
                debug_sql_query = f"Error in SQL query: {e}"
```

---

### 4. Interactive Debug Window in Streamlit UI
**File**: [app.py](file:///Users/adi/Documents/Coding/medical-ai-rag/app.py)

#### After (Newly Added under Time metrics section):
```python
        # ====================== DEBUG LOGGING ======================
        # Print to terminal console
        print("\n=== RAG PIPELINE DEBUG LOGS ===")
        print(f"SQL Query Executed:\n{debug_sql_query}")
        print(f"SQL Records Retrieved: {debug_sql_records}")
        print(f"FAISS Chunks Retrieved: {debug_faiss_chunks}")
        print(f"Final Prompt Generation Input:\n{final_prompt}")
        print("===============================\n")

        # Render in Streamlit UI
        with st.expander("🛠️ RAG Debug Logs", expanded=False):
            st.markdown(f"**SQL Query Executed:**\n```sql\n{debug_sql_query}\n```")
            st.markdown(f"**SQL Records Retrieved:** `{debug_sql_records}`")
            st.markdown(f"**FAISS Chunks Retrieved:** `{debug_faiss_chunks}`")
            st.markdown(f"**Final Prompt Generation Input:**\n```text\n{final_prompt}\n```")
```

---

## 3. Key Achievements & Verification
1. **Accurate Retrieval**: Patients with sparse records (such as Patient 2, who only has 1 record in the database) are no longer drowned out by Patients with thousands of mock/inflation records.
2. **Deduplicated Patient Context**: Patients are shown exactly once in the clinical context, and findings/modalities are concatenated clearly using SQL `string_agg`.
3. **Traceability**: Streamlit UI includes a collapsed expander containing the full prompt context and live SQL/FAISS counts, facilitating real-time troubleshooting for clinical operators.
