from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import time

# ====================== Setup ======================
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Load existing FAISS index
try:
    vectorstore = FAISS.load_local("./faiss_db", embeddings, allow_dangerous_deserialization=True)
except Exception as e:
    print("❌ Failed to load FAISS index. Did you run embed.py?", e)
    exit(1)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 10}
)

llm = ChatOllama(
    model="phi3",
    temperature=0.2,
    num_ctx=2048,
)

# Enhanced Prompt Template for Clinical EHR Processing
prompt_template = ChatPromptTemplate.from_template("""
You are a highly capable AI medical assistant interacting with EHR (Electronic Health Record) data.

[ PATIENT LAB EVENT CONTEXT ]
{context}

[ STRICT RULES ]
1. Answer using ONLY natural language. NEVER output raw SQL queries or database code.
2. Focus on clinical summarization, specifically pointing out any values flagged as ABNORMAL.
3. Provide cohesive patient insights based on the retrieved lab events.
4. Output ONLY the final analytical answer. DO NOT explain your reasoning.
5. If the answer cannot be confidently deduced from the Context, output exactly: "Not found in database".

Question: {question}

Final Clinical Answer:
""")

print("✅ FAISS-Backed Dynamic RAG System Ready!\n")

# ====================== Query Loop ======================
while True:
    query = input("\nAsk your clinical EHR question (or type 'exit'): ").strip()
    
    if query.lower() in ['exit', 'quit', 'bye']:
        print("Goodbye!")
        break
        
    if not query:
        continue

    start_time = time.time()
    docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = prompt_template.format(context=context, question=query)
    
    print("\nAnalyzing Patient Records...", end=" ")
    response = llm.invoke(prompt)
    
    end_time = time.time()

    print(f"\n\n🩺 AI Findings ({(end_time - start_time):.1f}s):\n")
    print(response.content if hasattr(response, 'content') else response)
    print("\n" + "-"*80)