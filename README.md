# RAG Evaluation Demo

This folder documents the RAG (Retrieval-Augmented Generation) evaluation script defined in `app.py` at the root of the repository. The script builds a small in-memory knowledge base, runs RAG over it using LangChain + Chroma, and then evaluates both retrieval quality and answer quality.

---
 
## YouTube Demo

Watch the full walkthrough and explanation of this project here:

https://youtu.be/5syl6THrTGI

---

## 1. High-level overview of `app.py`

`app.py` is a self-contained script that:

- **Loads configuration** from environment variables (API keys).
- **Builds a vector store** using synthetic HR/policy documents.
- **Constructs a LangChain RAG pipeline** (retriever → prompt → LLM → string output).
- **Defines an evaluation dataset** with questions, ground-truth answers, and relevant document IDs.
- **Computes retrieval metrics** (MRR, NDCG, MAP) for the retriever.
- **Computes generation metrics** (accuracy, completeness, conciseness) using an LLM-as-a-judge pattern.
- **Cleans up** the Chroma collection when finished.

### 1.1. Setup & configuration

Key imports and setup:

- Uses `python-dotenv` to load a local `.env` file.
- Requires two environment variables:
  - `GOOGLE_API_KEY` – used by `GoogleGenerativeAIEmbeddings` to build embeddings.
  - `GROQ_API_KEY` – used by `ChatGroq` to access Groq-hosted Llama models.
- Initializes three Groq LLMs:
  - `llama_3_1 = ChatGroq(model="llama-3.1-8b-instant", temperature=0)`
  - `llama_3_3 = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)`
  - `llama_4 = ChatGroq(model="meta-llama/llama-4-maverick-17b-128e-instruct", temperature=0)`
- Chooses `llm = llama_4` as the active model for both answering questions and acting as the judge.
- Initializes `gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")` for embedding documents and queries.

### 1.2. Knowledge base / documents

`documents` is a Python list of `langchain_core.documents.Document` objects. Each document:

- Contains a short synthetic company policy section (`page_content`).
- Has metadata with `chunk_id` (e.g. `"Chunk_1"`) and a `topic` string.

Topics include:

- Remote work policy
- Annual leave/time off
- Expense reimbursement
- Performance reviews
- Work-from-abroad policy
- Equipment policy
- Training budget
- Overtime compensation
- Health benefits
- Travel policy

These are the “ground truth” chunks used both for retrieval and for evaluation.

### 1.3. Vector store and retriever

The script builds a Chroma vector store in memory:

- `vectorstore = Chroma.from_documents(documents=documents, embedding=gemini_embeddings, collection_name="gemini_rag_test")`
- Then exposes a retriever:
  - `retriever = vectorstore.as_retriever(search_kwargs={"k": 2})`
  - It retrieves the **top 2** most similar chunks for each query.

### 1.4. RAG pipeline

The RAG chain is constructed as a LangChain `Runnable` pipeline:

1. **Prompt template**:
   ```text
   Answer the question based only on the following context:
   {context}

   Question: {question}
   ```
2. **Formatter**:
   - `format_docs(docs)` joins retrieved document texts with double newlines.
3. **Pipeline**:
   - `{"context": retriever | format_docs, "question": RunnablePassthrough()}`
   - `| prompt`
   - `| llm`
   - `| StrOutputParser()`

Invoking `rag_chain.invoke(question)`:

- Runs retrieval for the question.
- Formats context into the prompt.
- Calls the LLM to generate an answer.
- Returns a plain string.

### 1.5. Evaluation dataset

`eval_dataset` is a list of dictionaries, each with:

- `"question"`: a natural language query about company policies.
- `"ground_truth"`: the correct reference answer.
- `"relevant_chunk_id"`: the `chunk_id` of the document that contains the answer.

This dataset is used for both **retrieval evaluation** (does the retriever return the correct chunk?) and **generation evaluation** (does the generated answer match the ground truth?).

### 1.6. Retrieval metrics (MRR, NDCG, MAP)

Function: `calculate_retrieval_metrics(dataset, retriever)`

For each item in `eval_dataset`:

- Invokes the retriever with the question.
- Collects retrieved `chunk_id`s.
- Checks where the `relevant_chunk_id` appears (if at all).
- Computes per-query scores:
  - **MRR (Mean Reciprocal Rank)** – `1/rank` for the correct chunk, or 0 if not retrieved.
  - **NDCG (Normalized Discounted Cumulative Gain)** – rank-based score with log discounting.
  - **AP (Average Precision)** – with one relevant doc, this equals `1/rank` when retrieved.

After processing all questions, it prints and returns:

- Overall `mrr`
- Overall `ndcg`
- Overall `map` (mean average precision)

### 1.7. Generation evaluation (LLM-as-a-judge)

#### 1.7.1. Score schema

`GenerationEvalScores` is a `pydantic.BaseModel` with:

- `accuracy: float`
- `completeness: float`
- `conciseness: float`

This defines a structured schema for the judge’s response.

#### 1.7.2. Evaluation logic

Function: `evaluate_generation_with_gemini(dataset, rag_chain)`

For each example in `eval_dataset`:

1. Calls `rag_chain.invoke(question)` to produce the **student answer**.
2. Builds an **evaluation prompt** that includes:
   - Question
   - Ground truth answer
   - Student answer
   - Instructions to return 3 scores between 0 and 1 for:
     - accuracy
     - completeness
     - conciseness
   - `format_instructions` from `PydanticOutputParser` so the model returns a parseable JSON-like structure.
3. Sends the evaluation prompt to `llm.invoke(...)` (the same Groq Llama model).
4. Parses the response into a `GenerationEvalScores` instance.
5. Aggregates per-query scores into lists.

At the end, it computes averages:

- `accuracy`
- `completeness`
- `conciseness`

and prints them as overall generation quality metrics.

### 1.8. Main entry point

When you run `python app.py`, the script executes:

1. `calculate_retrieval_metrics(eval_dataset, retriever)`
2. `evaluate_generation_with_gemini(eval_dataset, rag_chain)`
3. `vectorstore.delete_collection()` to clean up the Chroma collection.

---

## 2. Dependencies (`requirements.txt`)

The root `requirements.txt` contains:

- `langchain-google-genai` – Gemini/Google Generative AI integrations for embeddings.
- `langchain-chroma` – Chroma integration for LangChain vector stores.
- `langchain-community` – Community integrations and utilities for LangChain.
- `langchain-core` – Core LangChain primitives (runnables, prompts, documents, etc.).
- `chromadb` – Chroma vector database backend.
- `numpy` – Used for numerical operations and averaging metrics.
- `python-dotenv` – Loads environment variables from a `.env` file.
- `langchain-groq` – Groq LLM integration.
- `groq` – Underlying Groq Python client.

You may additionally need the Google and Groq SDK dependencies that these packages pull in transitively. Installing from `requirements.txt` will handle that for most environments.

---

## 3. Environment variables and API keys

Before running `app.py`, you must set API keys for:

- **Google Generative AI** (for embeddings)
- **Groq** (for Llama models)

The script expects these in a `.env` file in the project root:

```bash
GOOGLE_API_KEY="your-google-api-key"
GROQ_API_KEY="your-groq-api-key"
```

The call to `load_dotenv()` in `app.py` will read this file and populate the environment.

> **Security note**: Do not commit `.env` with real keys to version control. Use local `.env` for development and secrets managers in production.

---

## 4. Reproducible steps to run the evaluation

Follow these steps to reproduce the RAG evaluation locally.

### 4.1. Clone the repository

```bash
git clone <your-repo-url>
cd YouTube
```

(Adjust the folder name if your repo uses a different root directory.)

### 4.2. Create and activate a virtual environment (recommended)

Using `venv` (Python 3):

```bash
python -m venv .venv
# On Windows PowerShell
. .venv\Scripts\Activate.ps1
# Or on cmd
.venv\Scripts\activate.bat
```

### 4.3. Install dependencies

From the project root (where `requirements.txt` is located):

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs LangChain, Chroma, Google GenAI, Groq, and other supporting libraries.

### 4.4. Configure API keys

Create a `.env` file at the project root:

```bash
GOOGLE_API_KEY="your-google-api-key"
GROQ_API_KEY="your-groq-api-key"
```

Make sure the keys are valid and have access to:

- Google Generative AI embeddings API (`models/text-embedding-004`).
- Groq-hosted Llama models (e.g., `meta-llama/llama-4-maverick-17b-128e-instruct`).

### 4.5. Run the evaluation script

From the project root:

```bash
python app.py
```

You should see console output similar to:

- For each question:
  - The question text
  - The expected `relevant_chunk_id`
  - The list of retrieved chunk IDs
- Overall retrieval scores:
  - `MRR`
  - `NDCG`
  - `MAP`
- Per-question generation evaluation:
  - Parsed scores for `Accuracy`, `Completeness`, `Conciseness`
- Overall average generation scores for each dimension.

If everything is configured correctly, the script will finish by deleting the Chroma collection.

---

## 5. How to extend or customize

Some ideas for extending this evaluation setup:

- **Add more documents**: Append more `Document` objects to the `documents` list to simulate a larger knowledge base.
- **Modify `k` in the retriever**: Change `search_kwargs={"k": 2}` to retrieve more or fewer documents.
- **Change the model**: Swap `llm` to one of the other defined Groq models (`llama_3_1` or `llama_3_3`) or another supported model.
- **Expand the evaluation dataset**: Add more questions, ground truths, and `relevant_chunk_id`s to better stress-test retrieval and generation.
- **Experiment with prompts**: Adjust the RAG prompt template to explore how instruction phrasing affects answer quality.

This README should give you a clear understanding of how `app.py` works and how `requirements.txt` supports it, plus the exact steps needed to reproduce the evaluation on your machine.
