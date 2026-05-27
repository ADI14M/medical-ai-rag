import json
import time
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import re

# ====================== Configuration ======================
RAG_MODEL = "phi3"
# For the Judge, it's highly recommended to use a stronger model (like llama3, mistral, or phi3)
# If you only have tinyllama installed, it will try its best but the evaluation scores might be inconsistent.
JUDGE_MODEL = "phi3" 
EMBED_MODEL = "nomic-embed-text"
TEST_DATA_PATH = "test_dataset.json"

# ====================== Setup RAG ======================
print(f"Loading Embeddings ({EMBED_MODEL})...")
embeddings = OllamaEmbeddings(model=EMBED_MODEL)

try:
    vectorstore = FAISS.load_local("./faiss_db", embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    print("FAISS index loaded successfully.")
except Exception as e:
    print("❌ Failed to load FAISS index. Did you run embed.py?", e)
    exit(1)

# RAG Generator
generator_llm = ChatOllama(model=RAG_MODEL, temperature=0.1, num_ctx=2048)

# RAG Prompt
rag_prompt = ChatPromptTemplate.from_template("""
You are a direct, robotic medical assistant.
[ PATIENT LAB EVENT CONTEXT ]
{context}

Question: {question}
Final Clinical Answer:
""")

# ====================== Setup Evaluator ======================
judge_llm = ChatOllama(model=JUDGE_MODEL, temperature=0.0)

judge_prompt = ChatPromptTemplate.from_template("""
You are an expert AI evaluator. You will evaluate a medical RAG system's answer based on the provided context and ground truth.
Evaluate on three metrics (Scale 1 to 5):
1. Faithfulness: Is the generated answer based ONLY on the provided context (no hallucination)?
2. Answer Relevance: Does the generated answer directly address the user's question?
3. Correctness: Does the generated answer match the factual intent of the ground truth?

Question: {question}
Retrieved Context: {context}
Ground Truth: {ground_truth}
Generated Answer: {generated_answer}

Provide your evaluation in the following strict JSON format:
{{
    "faithfulness": <score 1-5>,
    "answer_relevance": <score 1-5>,
    "correctness": <score 1-5>,
    "feedback": "<brief explanation>"
}}
""")

def extract_json(text):
    try:
        # Try to find a JSON block in the text
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(text)
    except Exception:
        return {"faithfulness": 0, "answer_relevance": 0, "correctness": 0, "feedback": "Failed to parse JSON."}

# ====================== Run Evaluation ======================
def run_evaluation():
    try:
        with open(TEST_DATA_PATH, "r") as f:
            test_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Could not find {TEST_DATA_PATH}")
        return

    print(f"\nStarting Evaluation on {len(test_data)} samples...")
    print(f"Generator Model: {RAG_MODEL} | Judge Model: {JUDGE_MODEL}\n")

    results = []
    
    for i, item in enumerate(test_data):
        print(f"--- Sample {i+1} ---")
        question = item['question']
        ground_truth = item['ground_truth']
        print(f"Q: {question}")
        
        # 1. Retrieve
        docs = retriever.invoke(question)
        context = "\n".join([d.page_content for d in docs])
        
        # 2. Generate
        prompt = rag_prompt.format(context=context, question=question)
        generated_answer = generator_llm.invoke(prompt).content
        print(f"Generated Answer: {generated_answer.strip()}")
        
        # 3. Evaluate
        eval_prompt = judge_prompt.format(
            question=question,
            context=context,
            ground_truth=ground_truth,
            generated_answer=generated_answer
        )
        eval_response = judge_llm.invoke(eval_prompt).content
        
        # 4. Parse
        scores = extract_json(eval_response)
        print(f"Scores: Faithfulness: {scores.get('faithfulness')}/5 | Relevance: {scores.get('answer_relevance')}/5 | Correctness: {scores.get('correctness')}/5")
        print(f"Feedback: {scores.get('feedback')}\n")
        
        results.append({
            "question": question,
            "scores": scores
        })
        time.sleep(1) # small delay

    # Summary
    if results:
        avg_f = sum([r['scores'].get('faithfulness', 0) for r in results]) / len(results)
        avg_r = sum([r['scores'].get('answer_relevance', 0) for r in results]) / len(results)
        avg_c = sum([r['scores'].get('correctness', 0) for r in results]) / len(results)
        
        print("=== EVALUATION SUMMARY ===")
        print(f"Average Faithfulness:     {avg_f:.2f}/5.00")
        print(f"Average Answer Relevance: {avg_r:.2f}/5.00")
        print(f"Average Correctness:      {avg_c:.2f}/5.00")

if __name__ == "__main__":
    run_evaluation()
