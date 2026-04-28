import psycopg2
from config import DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT

conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD, port=DB_PORT)
cur = conn.cursor()

# Get some stats
cur.execute("""
    SELECT p.patient_id, p.full_name, p.gender, s.study_date, s.priority, i.image_type, a.findings_summary, a.confidence_score 
    FROM oads.patients p 
    JOIN oads.studies s ON p.patient_id = s.patient_id 
    JOIN oads.images i ON s.study_id = i.study_id 
    JOIN oads.analysis a ON i.image_id = a.image_id 
    WHERE p.full_name = 'Ramesh' LIMIT 5;
""")
print("=== RAMESH DETAILS ===")
for row in cur.fetchall():
    print(row)
    
conn.close()
