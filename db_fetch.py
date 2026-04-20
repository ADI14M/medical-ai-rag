import psycopg2
from config import DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT

def fetch_lab_events():
    print("🔌 Connecting to database...")
    try:
        conn = psycopg2.connect(
            host=DB_HOST, 
            database=DB_NAME, 
            user=DB_USER, 
            password=DB_PASSWORD, 
            port=DB_PORT,
            connect_timeout=10
        )
        cur = conn.cursor()
        cur.execute("SELECT current_database();")
        print("Connected to:", cur.fetchone())
        
        # Inner join to combine patient demographics with lab data
        query = """
        SELECT 
            p.subject_id,
            p.gender,
            l.valuenum,
            l.valueuom,
            l.charttime,
            l.flag
        FROM ehr.patients p
        JOIN ehr.labevents l
        ON p.subject_id = l.subject_id::INTEGER
        WHERE l.valuenum IS NOT NULL
        LIMIT 1000;
        """
        
        print("📊 Fetching records...")
        cur.execute(query)
        rows = cur.fetchall()
        
        documents = []
        metadatas = []
        
        for row in rows:
            subject_id = row[0]
            gender = row[1] or "Unknown"
            valuenum = row[2]
            valueuom = row[3] or "units"
            charttime = row[4]
            flag = row[5]
            
            # Format: "Patient 101 (male) has lab value 145 mg/dL recorded at 2024-01-01"
            base_text = f"Patient {subject_id} ({gender}) has lab value {valuenum} {valueuom} recorded at {charttime}."
            
            if flag and flag.lower() == 'abnormal':
               base_text += " This value is flagged as ABNORMAL."
               
            documents.append(base_text)
            
            metadatas.append({
                "subject_id": subject_id,
                "gender": gender,
                "charttime": str(charttime),
                "is_abnormal": True if flag and flag.lower() == 'abnormal' else False
            })
            
        print(f"✅ Fetched and formatted {len(documents)} lab string events.")
        
        return documents, metadatas
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise
    finally:
        if 'cur' in locals() and cur:
            cur.close()
        if 'conn' in locals() and conn:
            conn.close()
            print("🔌 Database connection closed")

if __name__ == "__main__":
    docs, metas = fetch_lab_events()
    if docs:
        print("\n[ Example Chunk ]")
        print(docs[0])
