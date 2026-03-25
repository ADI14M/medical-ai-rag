from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import time

# ====================== Setup ======================
embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = Chroma(
    persist_directory="./vector_db",
    embedding_function=embeddings,
    collection_name="my_rag_collection"   # must match what you used while creating
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})   # increased to 4 for better context

# Use ChatOllama (better than old Ollama class)
llm = ChatOllama(
    model="phi3",           # or "phi3:medium" if you pulled the bigger one
    temperature=0.3,        # lower = more factual
    # num_ctx=4096,         # uncomment if you need larger context
)

# Better prompt template
prompt_template = ChatPromptTemplate.from_template("""
You are a helpful medical assistant. Answer the question using only the provided context.
If you don't know the answer based on the context, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:
""")

print("✅ RAG system ready! (Medical data loaded)\n")

# ====================== Query Loop ======================
while True:
    query = input("\nAsk your medical question (or type 'exit' to quit): ").strip()
    
    if query.lower() in ['exit', 'quit', 'bye']:
        print("Goodbye!")
        break
        
    if not query:
        continue

    start_time = time.time()

    # Retrieve relevant chunks
    docs = retriever.invoke(query)                    # modern way (instead of get_relevant_documents)

    context = "\n\n".join([doc.page_content for doc in docs])

    # Generate response
    prompt = prompt_template.format(context=context, question=query)
    
    print("\nThinking...", end=" ")
    response = llm.invoke(prompt)
    
    end_time = time.time()

    print(f"\n\nAI Response ({(end_time - start_time):.1f}s):\n")
    print(response.content if hasattr(response, 'content') else response)
    
    print("\n" + "-"*80)