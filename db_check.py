import psycopg2
from config import DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT

conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD, port=DB_PORT)
cur = conn.cursor()

for table in ['patients', 'studies', 'images', 'analysis']:
    cur.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_schema = 'oads' AND table_name = '{table}';")
    print(f"\n--- {table.upper()} ---")
    for row in cur.fetchall():
        print(f"  {row[0]}: {row[1]}")
