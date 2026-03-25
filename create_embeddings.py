from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from extract_data import extract_medical_documents
import time
from tqdm import tqdm
import math

# ====================== LOAD DATA ======================
print("📥 Extracting data from PostgreSQL...")

data = extract_medical_documents()

documents = data["documents"]
metadatas = data["metadatas"]
ids = data["ids"]

print(f"✅ Extracted Documents: {len(documents)}")

# ====================== EMBEDDINGS ======================
embeddings = OllamaEmbeddings(model="nomic-embed-text")

batch_size = 64
total_docs = len(documents)

start_time = time.time()

vectorstore = Chroma(
    persist_directory="./vector_db",
    embedding_function=embeddings,
    collection_name="my_rag_collection"
)

# ====================== ADD WITH METADATA (CRITICAL FIX) ======================
for i in tqdm(range(0, total_docs, batch_size), desc="Embedding Progress"):
    batch_docs = documents[i:i+batch_size]
    batch_meta = metadatas[i:i+batch_size]
    batch_ids = ids[i:i+batch_size]

    vectorstore.add_texts(
        texts=batch_docs,
        metadatas=batch_meta,
        ids=batch_ids
    )

# ====================== SUMMARY ======================
end_time = time.time()

print("\n" + "="*60)
print("✅ EMBEDDING COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"Total stored: {vectorstore._collection.count()}")
print(f"Time taken: {(end_time - start_time)/60:.2f} minutes")
print("="*60)