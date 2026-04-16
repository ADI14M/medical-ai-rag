import psycopg2
from config import DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT

try:
    conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD, port=DB_PORT)
    cur = conn.cursor()
    cur.execute("""
        SELECT table_name, column_name, data_type 
        FROM information_schema.columns 
        WHERE table_schema = 'ehr' 
        ORDER BY table_name;
    """)
    for row in cur.fetchall():
        print(f"{row[0]} | {row[1]} : {row[2]}")
except Exception as e:
    print('Error:', e)
