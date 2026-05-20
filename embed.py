import os
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from extract_data import extract_medical_documents
import time

def create_embeddings():
    print("\n[ Phase 1 ] Fetching Medical EHR Data...")
    data = extract_medical_documents()
    docs = data["documents"]
    metadatas = data["metadatas"]
    
    if not docs:
        print("No documents found in EHR to embed.")
        return
        
    print("\n[ Phase 2 ] Computing Vector Embeddings & Building FAISS Index...")
    start_time = time.time()
    
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    vectorstore = FAISS.from_texts(
        texts=docs,
        embedding=embeddings,
        metadatas=metadatas
    )
    
    # Save the FAISS index to the local directory
    vectorstore.save_local("./faiss_db")
    
    end_time = time.time()
    
    print("\n" + "="*60)
    print(f"✅ FAISS RAG COMPILATION COMPLETED SUCCESSFULLY!")
    print(f"Vectors Computed: {vectorstore.index.ntotal}")
    print(f"Time taken: {(end_time - start_time):.2f} seconds")
    print("="*60)

if __name__ == "__main__":
    create_embeddings()
