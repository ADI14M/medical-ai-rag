import psycopg2
from config import DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT
from tqdm import tqdm
import hashlib
import pickle
from datetime import datetime

print("🔌 Connecting to database...")

def extract_medical_documents():
    conn = None
    cursor = None

    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT,
            connect_timeout=10
        )

        cursor = conn.cursor()

        # ✅ OPTIMIZED QUERY with missing details
        query = """
        SELECT 
            p.patient_id,
            p.full_name,
            p.gender,
            s.study_date,
            s.priority,
            i.image_type,
            a.findings_summary,
            a.confidence_score
        FROM oads.patients p
        JOIN oads.studies s ON p.patient_id = s.patient_id
        JOIN oads.images i ON s.study_id = i.study_id
        JOIN oads.analysis a ON i.image_id = a.image_id
        ORDER BY s.study_date DESC;
        """

        print("📊 Fetching records in chunks...")
        cursor.execute(query)

        documents = []
        metadatas = []
        ids = []

        seen = set()
        total_rows = 0
        
        while True:
            # ✅ OPTIMIZATION: fetchmany instead of fetchall
            rows = cursor.fetchmany(1000)
            if not rows:
                break
                
            total_rows += len(rows)

            for row in rows:

                # ✅ UPDATED COLUMN MAPPING
                patient_id = row[0]
                full_name = row[1]
                gender = row[2] or "Unknown"
                study_date = row[3]
                priority = row[4] or "Normal"
                image_type = row[5]
                findings = row[6] or "No findings recorded"
                confidence = row[7] if row[7] is not None else 0.0

                # Normalize date
                if isinstance(study_date, datetime):
                    study_date = study_date.strftime("%Y-%m-%d")

                # ✅ DEDUP KEY (fixed to include formatted confidence)
                confidence_str = f"{confidence:.2f}"
                key_str = f"{patient_id}|{study_date}|{image_type}|{findings}|{confidence_str}"
                key_hash = hashlib.md5(key_str.encode()).hexdigest()

                if key_hash in seen:
                    continue
                seen.add(key_hash)

                # ✅ DOCUMENT TEXT (important for embeddings)
                text = f"""Patient ID: {patient_id}
Patient Name: {full_name}
Gender: {gender}
Study Date: {study_date}
Study Priority: {priority}
Image Type: {image_type}
Findings: {findings}
AI Confidence Score: {confidence:.2f}
"""

                documents.append(text)

                # ✅ CLEAN METADATA (CRITICAL FOR FILTERING)
                metadatas.append({
                    "patient_id": int(patient_id),
                    "gender": gender,
                    "priority": priority,
                    "study_date": study_date,
                    "image_type": image_type,
                    "source": "oads_db"
                })

                # Stable ID (based on hash → prevents duplication across runs)
                ids.append(key_hash)

        print(f"\n✅ Total rows fetched: {total_rows}")
        print(f"✅ Final unique documents: {len(documents)}")
        print(f"   Removed duplicates: {total_rows - len(documents)}")

        data = {
            "documents": documents,
            "metadatas": metadatas,
            "ids": ids
        }

        # ✅ SAVE FILE
        with open("processed_data.pkl", "wb") as f:
            pickle.dump(data, f)

        print("💾 Saved to processed_data.pkl")

        return data

    except Exception as e:
        print(f"❌ Error: {e}")
        raise

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
            print("🔌 Database connection closed")


# ====================== RUN ======================
if __name__ == "__main__":
    extract_medical_documents()