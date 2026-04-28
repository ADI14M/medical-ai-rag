import psycopg2
import random
from datetime import datetime, timedelta
from config import DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT

def populate_mock_data():
    conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD, port=DB_PORT)
    cur = conn.cursor()
    
    # Get patients without studies
    cur.execute("""
        SELECT p.patient_id 
        FROM oads.patients p
        LEFT JOIN oads.studies s ON p.patient_id = s.patient_id
        WHERE s.study_id IS NULL
    """)
    patients_to_populate = [row[0] for row in cur.fetchall()]
    
    print(f"Found {len(patients_to_populate)} patients without data. Populating...")
    
    image_types = ['xray', 'ct', 'mri', 'ultrasound']
    priorities = ['low', 'medium', 'high', 'critical']
    findings = [
        "Normal study, no abnormalities detected.",
        "Mild inflammation observed.",
        "Suspicious nodule detected, recommend follow-up.",
        "Clear signs of fracture.",
        "Degenerative changes consistent with age.",
        "Fluid accumulation noted.",
        "Unremarkable findings.",
        "Minor calcifications present."
    ]
    
    count = 0
    try:
        for pid in patients_to_populate:
            # Generate random date in past 2 years
            days_ago = random.randint(1, 730)
            study_date = datetime.now() - timedelta(days=days_ago)
            priority = random.choice(priorities)
            
            # Insert Study
            cur.execute("""
                INSERT INTO oads.studies (patient_id, study_date, priority)
                VALUES (%s, %s, %s)
                RETURNING study_id
            """, (pid, study_date, priority))
            study_id = cur.fetchone()[0]
            
            # Insert Image
            img_type = random.choice(image_types)
            file_path = f"/data/images/{img_type}_{study_id}.dcm"
            cur.execute("""
                INSERT INTO oads.images (study_id, image_type, file_path)
                VALUES (%s, %s, %s)
                RETURNING image_id
            """, (study_id, img_type, file_path))
            image_id = cur.fetchone()[0]
            
            # Insert Analysis
            conf_score = round(random.uniform(0.60, 0.99), 2)
            finding = random.choice(findings)
            cur.execute("""
                INSERT INTO oads.analysis (image_id, confidence_score, findings_summary)
                VALUES (%s, %s, %s)
            """, (image_id, conf_score, finding))
            
            count += 1
            if count % 500 == 0:
                conn.commit()
                print(f"Populated {count} patients...")
                
        conn.commit()
        print(f"Successfully populated {count} patients with random data!")
        
    except Exception as e:
        conn.rollback()
        print(f"Error: {e}")
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    populate_mock_data()
