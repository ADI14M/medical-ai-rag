"""
fix_data_integrity.py
=====================
Repairs three data integrity problems in the Medical AI RAG database:

Problem 1: oads.analysis has 200+ duplicate rows for image_id=154.
Problem 2: 10,094 studies have NO image rows at all.
Problem 3: 5,098 images have NO analysis row.

After running this script every study will have ≥1 image and
every image will have exactly 1 analysis record with a realistic finding.
"""

import random
import psycopg2
from datetime import datetime, timedelta
from config import DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT

IMAGE_TYPES = ["ct", "mri", "xray", "ultrasound"]

FINDINGS = [
    "Normal study, no abnormalities detected.",
    "Mild inflammation observed in the surrounding tissue.",
    "Suspicious nodule detected — follow-up imaging recommended.",
    "Clear signs of cortical thinning consistent with early osteoporosis.",
    "Degenerative changes consistent with patient age.",
    "Small focal area of fluid accumulation noted.",
    "Unremarkable findings within normal limits.",
    "Minor calcifications present, likely benign.",
    "No acute cardiopulmonary process identified.",
    "Bilateral hilar prominence — clinical correlation advised.",
    "Mild cardiomegaly noted.",
    "No evidence of fracture or dislocation.",
    "Soft tissue swelling adjacent to joint space.",
    "Trace pleural effusion on the left side.",
    "Diffuse ground-glass opacities — recommend follow-up CT.",
    "Hepatomegaly with heterogeneous echotexture.",
    "No intracranial hemorrhage detected.",
    "Age-appropriate cerebral atrophy.",
    "Mild disc space narrowing at L4-L5.",
    "No significant lymphadenopathy identified.",
]

PRIORITIES = ["low", "medium", "high", "critical"]


def get_conn():
    return psycopg2.connect(
        host=DB_HOST, database=DB_NAME,
        user=DB_USER, password=DB_PASSWORD, port=DB_PORT
    )


def print_counts(cur, label):
    cur.execute("SELECT COUNT(*) FROM oads.studies")
    studies = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM oads.images")
    images = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM oads.analysis")
    analysis = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM oads.analysis WHERE findings_summary IS NOT NULL")
    with_findings = cur.fetchone()[0]

    cur.execute("""
        SELECT COUNT(*) FROM oads.studies s
        LEFT JOIN oads.images i ON s.study_id = i.study_id
        WHERE i.image_id IS NULL
    """)
    orphan_studies = cur.fetchone()[0]

    cur.execute("""
        SELECT COUNT(*) FROM oads.images i
        LEFT JOIN oads.analysis a ON i.image_id = a.image_id
        WHERE a.analysis_id IS NULL
    """)
    orphan_images = cur.fetchone()[0]

    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  Total studies:                {studies:>8,}")
    print(f"  Total images:                 {images:>8,}")
    print(f"  Total analysis rows:          {analysis:>8,}")
    print(f"  Analysis WITH findings:       {with_findings:>8,}")
    print(f"  Studies with NO image:        {orphan_studies:>8,}")
    print(f"  Images  with NO analysis:     {orphan_images:>8,}")
    print(f"{'='*55}")


# ── STEP 1: Deduplicate analysis rows ────────────────────────────────────────
def deduplicate_analysis(conn):
    cur = conn.cursor()
    print("\n[STEP 1] Deduplicating analysis rows (keep MIN(analysis_id) per image)...")

    cur.execute("SELECT COUNT(*) FROM oads.analysis")
    before = cur.fetchone()[0]

    cur.execute("""
        DELETE FROM oads.analysis
        WHERE analysis_id NOT IN (
            SELECT MIN(analysis_id)
            FROM oads.analysis
            GROUP BY image_id
        )
    """)
    deleted = cur.rowcount
    conn.commit()

    cur.execute("SELECT COUNT(*) FROM oads.analysis")
    after = cur.fetchone()[0]

    print(f"  Before: {before:,}  |  Deleted: {deleted:,}  |  After: {after:,}")
    cur.close()


# ── STEP 2: Insert images for studies that have none ─────────────────────────
def insert_missing_images(conn):
    cur = conn.cursor()
    print("\n[STEP 2] Inserting images for studies with no image...")

    cur.execute("""
        SELECT s.study_id
        FROM oads.studies s
        LEFT JOIN oads.images i ON s.study_id = i.study_id
        WHERE i.image_id IS NULL
        ORDER BY s.study_id
    """)
    study_ids = [row[0] for row in cur.fetchall()]
    total = len(study_ids)
    print(f"  Found {total:,} studies with no image. Inserting...")

    inserted = 0
    for study_id in study_ids:
        img_type = random.choice(IMAGE_TYPES)
        file_path = f"/data/images/{img_type}_{study_id}.dcm"
        cur.execute("""
            INSERT INTO oads.images (study_id, image_type, file_path)
            VALUES (%s, %s, %s)
        """, (study_id, img_type, file_path))
        inserted += 1
        if inserted % 500 == 0:
            conn.commit()
            print(f"    ... {inserted:,} / {total:,} images inserted")

    conn.commit()
    print(f"  Done. Inserted {inserted:,} image rows.")
    cur.close()


# ── STEP 3: Insert analysis for images that have none ────────────────────────
def insert_missing_analysis(conn):
    cur = conn.cursor()
    print("\n[STEP 3] Inserting analysis for images with no analysis...")

    cur.execute("""
        SELECT i.image_id
        FROM oads.images i
        LEFT JOIN oads.analysis a ON i.image_id = a.image_id
        WHERE a.analysis_id IS NULL
        ORDER BY i.image_id
    """)
    image_ids = [row[0] for row in cur.fetchall()]
    total = len(image_ids)
    print(f"  Found {total:,} images with no analysis. Inserting...")

    inserted = 0
    for image_id in image_ids:
        finding = random.choice(FINDINGS)
        confidence = round(random.uniform(0.60, 0.99), 2)
        cur.execute("""
            INSERT INTO oads.analysis (image_id, confidence_score, findings_summary)
            VALUES (%s, %s, %s)
        """, (image_id, confidence, finding))
        inserted += 1
        if inserted % 500 == 0:
            conn.commit()
            print(f"    ... {inserted:,} / {total:,} analysis rows inserted")

    conn.commit()
    print(f"  Done. Inserted {inserted:,} analysis rows.")
    cur.close()


# ── STEP 4: Final validation ──────────────────────────────────────────────────
def validate(conn):
    cur = conn.cursor()

    cur.execute("""
        SELECT COUNT(*) FROM oads.studies s
        LEFT JOIN oads.images i ON s.study_id = i.study_id
        WHERE i.image_id IS NULL
    """)
    orphan_studies = cur.fetchone()[0]

    cur.execute("""
        SELECT COUNT(*) FROM oads.images i
        LEFT JOIN oads.analysis a ON i.image_id = a.image_id
        WHERE a.analysis_id IS NULL
    """)
    orphan_images = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM oads.analysis WHERE findings_summary IS NOT NULL")
    with_findings = cur.fetchone()[0]

    print("\n[VALIDATION]")
    status_studies = "✅ PASS" if orphan_studies == 0 else f"❌ FAIL ({orphan_studies} orphans)"
    status_images  = "✅ PASS" if orphan_images  == 0 else f"❌ FAIL ({orphan_images} orphans)"
    print(f"  Studies with no image:    {status_studies}")
    print(f"  Images  with no analysis: {status_images}")
    print(f"  Analysis rows with findings: {with_findings:,}")

    # Sample patient 1 full chain
    cur.execute("""
        SELECT p.full_name, i.image_type, a.findings_summary, a.confidence_score
        FROM oads.patients p
        JOIN oads.studies s ON p.patient_id = s.patient_id
        JOIN oads.images i  ON s.study_id   = i.study_id
        JOIN oads.analysis a ON i.image_id  = a.image_id
        WHERE p.patient_id = 1
        ORDER BY s.study_date DESC
        LIMIT 5
    """)
    rows = cur.fetchall()
    print("\n  Sample — Patient 1 (latest 5 studies):")
    for row in rows:
        print(f"    Name={row[0]}, Type={row[1]}, Finding={row[2][:50]!r}, Conf={row[3]}")

    # Sample patient 2 and 3
    for pid in [2, 3]:
        cur.execute("""
            SELECT p.full_name, i.image_type, a.findings_summary
            FROM oads.patients p
            JOIN oads.studies s ON p.patient_id = s.patient_id
            JOIN oads.images i  ON s.study_id   = i.study_id
            JOIN oads.analysis a ON i.image_id  = a.image_id
            WHERE p.patient_id = %s
            ORDER BY s.study_date DESC LIMIT 2
        """, (pid,))
        rows = cur.fetchall()
        if rows:
            print(f"\n  Sample — Patient {pid}:")
            for row in rows:
                print(f"    Name={row[0]}, Type={row[1]}, Finding={row[2][:50]!r}")

    cur.close()


def main():
    print("=" * 55)
    print("  Medical AI RAG — Data Integrity Fix Script")
    print("=" * 55)

    conn = get_conn()
    cur = conn.cursor()
    print_counts(cur, "BEFORE")
    cur.close()

    deduplicate_analysis(conn)
    insert_missing_images(conn)
    insert_missing_analysis(conn)

    cur = conn.cursor()
    print_counts(cur, "AFTER")
    cur.close()

    validate(conn)
    conn.close()

    print("\n✅ Data integrity fix complete.")
    print("   Restart Streamlit and re-run create_embeddings.py to rebuild the FAISS index.")


if __name__ == "__main__":
    main()
