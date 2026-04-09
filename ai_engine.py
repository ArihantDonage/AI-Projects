from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from rank_bm25 import BM25Okapi
import pickle
import os

# Initialize local embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load Vector DB
db = FAISS.load_local(
    "vector_db",
    embeddings,
    allow_dangerous_deserialization=True
)

# Load Keyword DB
with open("vector_db/docs.pkl", "rb") as f:
    docs = pickle.load(f)

texts = [d.page_content for d in docs]
tokenized = [t.split() for t in texts]
bm25 = BM25Okapi(tokenized)

# Initialize Groq LLM (Ensure GROQ_API_KEY is in your environment variables)
llm = ChatGroq(
    api_key=" ", # Add your key here
    temperature=0.1,
    model_name="llama-3.3-70b-versatile",
    max_tokens=2048
)

def retrieve_context(question):
    # Vector search
    vector_docs = db.similarity_search(question, k=4)

    # BM25 keyword search
    tokenized_query = question.split()
    scores = bm25.get_scores(tokenized_query)
    top_docs_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]
    keyword_docs = [texts[i] for i in top_docs_indices]

    # Combine context
    context = "\n\n".join([d.page_content for d in vector_docs] + keyword_docs)
    return context

def ask_network_ai(query, tool_mode, file_context=""):
    context = retrieve_context(query)
    
    system_prompt = """You are an expert Senior Network Automation Engineer.
    You specialize in routing protocols, network security, and Python automation.
    Provide highly technical, accurate, and structured answers. Do not hallucinate commands."""

    user_prompt = f"TASK MODE: {tool_mode}\n\n"
    if file_context:
        user_prompt += f"UPLOADED FILE CONTEXT:\n{file_context}\n\n"
    
    user_prompt += f"RETRIEVED KNOWLEDGE BASE CONTEXT:\n{context}\n\n"
    user_prompt += f"USER QUERY:\n{query}"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    # Stream the response back for the typewriter effect
    return llm.stream(messages)
