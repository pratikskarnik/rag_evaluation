import os
import math
import numpy as np
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================

# Load environment variables from .env
load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    print("⚠️ WARNING: Please set your GOOGLE_API_KEY in the .env file!")

if not os.getenv("GROQ_API_KEY"):
    print("⚠️ WARNING: Please set your GROQ_API_KEY in the .env file!")

# Initialize Gemini Models
# Using Groq for chat and 'text-embedding-004' for vectors
llama_3_1 = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
llama_3_3 = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
llama_4 = ChatGroq(model="meta-llama/llama-4-maverick-17b-128e-instruct", temperature=0)
llm=llama_4
gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# ==========================================
# 2. DATA PREPARATION (The Knowledge Base)
# ==========================================

# We manually create chunks and assign IDs so we can track them for MRR/NDCG
documents = [
    Document(
        page_content="""
        ## Section 1: Remote Work Policy
        Employees are allowed to work remotely up to 3 days a week. The core working hours 
        are 10:00 AM to 3:00 PM EST. Requests for full-time remote work must be approved 
        by a Director-level manager.
        """,
        metadata={"chunk_id": "Chunk_1", "topic": "Remote Work"}
    ),
    Document(
        page_content="""
        ## Section 2: Annual Leave and Time Off
        All full-time employees are entitled to 20 days of paid annual leave per year. 
        Unused leave can be carried over but is capped at 10 days; excess is forfeited 
        on January 1st.
        """,
        metadata={"chunk_id": "Chunk_2", "topic": "Leave"}
    ),
    Document(
        page_content="""
        ## Section 3: Expense Reimbursement
        TechCorp reimburses business meals capped at $50 per person. All expenses over 
        $20 require a valid receipt. Expense reports must be submitted within 30 days.
        """,
        metadata={"chunk_id": "Chunk_3", "topic": "Expenses"}
    ),
    Document(
        page_content="""
        ## Section 4: Performance Reviews
        Reviews are conducted bi-annually in June and December. Employees rated 4 or 5 
        are eligible for the annual bonus, calculated as 10% of base salary.
        """,
        metadata={"chunk_id": "Chunk_4", "topic": "Performance"}
    ),
    Document(
        page_content="""
        ## Section 5: Work From Abroad
        Employees may work from countries outside their home country for up to 30 days per year.
        All work-from-abroad requests must be approved by HR to ensure tax and compliance checks.
        """,
        metadata={"chunk_id": "Chunk_5", "topic": "Work From Abroad"}
    ),
    Document(
        page_content="""
        ## Section 6: Equipment Policy
        The company provides a standard laptop and one external monitor to all full-time employees.
        Additional equipment such as ergonomic chairs must be requested through the facilities portal.
        """,
        metadata={"chunk_id": "Chunk_6", "topic": "Equipment"}
    ),
    Document(
        page_content="""
        ## Section 7: Training Budget
        Each employee has an annual training budget of $1,000 for conferences, courses, and books.
        Any single expense above $500 requires manager approval before booking.
        """,
        metadata={"chunk_id": "Chunk_7", "topic": "Training"}
    ),
    Document(
        page_content="""
        ## Section 8: Overtime Compensation
        Non-exempt employees receive 1.5x pay for approved overtime hours worked beyond 40 hours per week.
        Overtime must be pre-approved by a manager and recorded in the time tracking system.
        """,
        metadata={"chunk_id": "Chunk_8", "topic": "Overtime"}
    ),
    Document(
        page_content="""
        ## Section 9: Health Benefits
        Health insurance coverage begins on the first day of the month following an employee's start date.
        The company covers 80% of the premium for employees and 50% for dependents.
        """,
        metadata={"chunk_id": "Chunk_9", "topic": "Health Benefits"}
    ),
    Document(
        page_content="""
        ## Section 10: Travel Policy
        All business trips must be booked through the approved travel portal.
        Economy class is required for flights under 6 hours; premium economy may be used for longer flights.
        """,
        metadata={"chunk_id": "Chunk_10", "topic": "Travel"}
    ),
]

# Build Vector Database
vectorstore = Chroma.from_documents(
    documents=documents, 
    embedding=gemini_embeddings,
    collection_name="gemini_rag_test"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) # Retrieve top 2

# ==========================================
# 3. THE RAG PIPELINE
# ==========================================

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ==========================================
# 4. EVALUATION DATASET (Ground Truth)
# ==========================================

eval_dataset = [
    {
        "question": "How many days can I work from home?",
        "ground_truth": "Employees are allowed to work remotely up to 3 days a week.",
        "relevant_chunk_id": "Chunk_1"
    },
    {
        "question": "What is the limit for meal expenses?",
        "ground_truth": "Meals are capped at $50 per person.",
        "relevant_chunk_id": "Chunk_3"
    },
    {
        "question": "When do I get my performance bonus?",
        "ground_truth": "Employees rated 4 or 5 are eligible after reviews in June and December.",
        "relevant_chunk_id": "Chunk_4"
    },
    {
        "question": "What happens to unused vacation days?",
        "ground_truth": "You can carry over 10 days, but any excess is forfeited on Jan 1st.",
        "relevant_chunk_id": "Chunk_2"
    },
    {
        "question": "How long can I work from another country each year?",
        "ground_truth": "You may work from abroad for up to 30 days per year, with HR approval.",
        "relevant_chunk_id": "Chunk_5"
    },
    {
        "question": "What equipment does the company provide by default?",
        "ground_truth": "A standard laptop and one external monitor are provided to all full-time employees.",
        "relevant_chunk_id": "Chunk_6"
    },
    {
        "question": "How much is my annual training budget?",
        "ground_truth": "Each employee has an annual training budget of $1,000.",
        "relevant_chunk_id": "Chunk_7"
    },
    {
        "question": "What overtime rate do non-exempt employees receive?",
        "ground_truth": "Non-exempt employees receive 1.5x pay for approved overtime beyond 40 hours per week.",
        "relevant_chunk_id": "Chunk_8"
    },
    {
        "question": "When does health insurance coverage begin?",
        "ground_truth": "Coverage begins on the first day of the month following the start date.",
        "relevant_chunk_id": "Chunk_9"
    },
    {
        "question": "What class can I book for a 5-hour business flight?",
        "ground_truth": "You must book economy class for flights under 6 hours.",
        "relevant_chunk_id": "Chunk_10"
    },
]

# ==========================================
# 5. RETRIEVAL METRICS (MRR & NDCG)
# ==========================================

def calculate_retrieval_metrics(dataset, retriever):
    print("\n--- 🔍 Evaluating Retrieval ---")
    reciprocal_ranks = []  # for MRR
    ndcg_scores = []       # for NDCG
    average_precisions = []  # for MAP
    hits = 0
    
    for item in dataset:
        q = item['question']
        target_id = item['relevant_chunk_id']
        
        # Get retrieved docs from Vector DB
        retrieved_docs = retriever.invoke(q)
        retrieved_ids = [doc.metadata.get("chunk_id") for doc in retrieved_docs]
        
        print(f"Q: {q}")
        print(f"   Expected: {target_id} | Retrieved: {retrieved_ids}")
        
        # Calculate ranking-based metrics (single relevant doc per query)
        if target_id in retrieved_ids:
            rank_position = retrieved_ids.index(target_id) + 1  # 1-based
            # MRR component
            rr = 1.0 / rank_position
            hits += 1

            # NDCG component (relevance = 1 for the single relevant doc)
            dcg = 1.0 / math.log2(rank_position + 1)
            idcg = 1.0  # ideal DCG when relevant doc is at rank 1
            ndcg = dcg / idcg

            # AP component (with one relevant doc, AP = precision at its rank)
            ap = 1.0 / rank_position
        else:
            rr = 0.0
            ndcg = 0.0
            ap = 0.0

        reciprocal_ranks.append(rr)
        ndcg_scores.append(ndcg)
        average_precisions.append(ap)
        
    mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
    mean_ndcg = float(np.mean(ndcg_scores)) if ndcg_scores else 0.0
    map_score = float(np.mean(average_precisions)) if average_precisions else 0.0

    print(f"\n✅ Retrieval Score (MRR):  {mrr:.2f} (1.0 is perfect)")
    print(f"✅ Retrieval Score (NDCG): {mean_ndcg:.2f} (1.0 is perfect)")
    print(f"✅ Retrieval Score (MAP):  {map_score:.2f} (1.0 is perfect)")

    return {"mrr": mrr, "ndcg": mean_ndcg, "map": map_score}

class GenerationEvalScores(BaseModel):
    """Structured scores for generation evaluation."""

    accuracy: float
    completeness: float
    conciseness: float

# ==========================================
# 6. GENERATION METRICS (LLM-as-a-Judge)
# ==========================================

def evaluate_generation_with_gemini(dataset, rag_chain):
    print("\n--- 🤖 Evaluating Generation (Accuracy, Completeness, Conciseness) ---")
    accuracy_scores = []
    completeness_scores = []
    conciseness_scores = []

    parser = PydanticOutputParser(pydantic_object=GenerationEvalScores)
    format_instructions = parser.get_format_instructions()
    
    for item in dataset:
        q = item['question']
        ground_truth = item['ground_truth']
        
        # 1. Get RAG Generation
        generated_answer = rag_chain.invoke(q)
        print(f"\nQ: {q}")
        print(f"\nGround Truth Answer: {ground_truth}")
        print(f"\nStudent Answer: {generated_answer}")
        # 2. Ask LLM Judge to grade the answer on multiple dimensions
        eval_prompt = f"""
        You are a strict teacher grading a student's answer on multiple dimensions.

        Question: {q}
        Ground Truth Answer: {ground_truth}
        Student Answer: {generated_answer}

        You must return three scores between 0 and 1 (inclusive):
        - accuracy: how factually correct and aligned with the ground truth the answer is.
        - completeness: how fully the answer covers the important points in the ground truth.
        - conciseness: how clear and to-the-point the answer is (without unnecessary fluff).

        Guidelines:
        - 1.0 means perfect for that dimension.
        - 0.5 means partially acceptable.
        - 0.0 means poor.

        {format_instructions}
        """
        
        try:
            score_response = llm.invoke(eval_prompt)
            parsed: GenerationEvalScores = parser.parse(score_response.content)
            acc = float(parsed.accuracy)
            comp = float(parsed.completeness)
            conc = float(parsed.conciseness)
            raw = score_response.content.strip()
        except Exception:
            acc = 0.0
            comp = 0.0
            conc = 0.0
            raw = "parse_error"

        accuracy_scores.append(acc)
        completeness_scores.append(comp)
        conciseness_scores.append(conc)

        
        # print(f"   Gen: {generated_answer}")
        # print(f"   Raw scores: {raw}")
        print(f"\nParsed -> Accuracy: {acc}, Completeness: {comp}, Conciseness: {conc}")
        
    avg_accuracy = float(np.mean(accuracy_scores)) if accuracy_scores else 0.0
    avg_completeness = float(np.mean(completeness_scores)) if completeness_scores else 0.0
    avg_conciseness = float(np.mean(conciseness_scores)) if conciseness_scores else 0.0

    print(f"\n✅ Generation Accuracy Score:      {avg_accuracy:.2f} / 1.0")
    print(f"✅ Generation Completeness Score: {avg_completeness:.2f} / 1.0")
    print(f"✅ Generation Conciseness Score: {avg_conciseness:.2f} / 1.0")

    return {
        "accuracy": avg_accuracy,
        "completeness": avg_completeness,
        "conciseness": avg_conciseness,
    }

# ==========================================
# 7. RUN EVERYTHING
# ==========================================

if __name__ == "__main__":
    # 1. Run Retrieval Eval
    calculate_retrieval_metrics(eval_dataset, retriever)
    
    # 2. Run Generation Eval
    evaluate_generation_with_gemini(eval_dataset, rag_chain)
    
    # Cleanup
    vectorstore.delete_collection()