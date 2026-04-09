from langchain_community.document_loaders import DirectoryLoader, TextLoader # Added TextLoader here
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import pickle
import os

print("Loading documents from docs/ directory...")
# Added loader_cls=TextLoader below
loader = DirectoryLoader("docs", glob="**/*.txt", loader_cls=TextLoader)
documents = loader.load()

print(f"Loaded {len(documents)} documents. Splitting text...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)
chunks = splitter.split_documents(documents)

print("Downloading/Loading embedding model (this runs locally)...")
# Using a fast, lightweight local embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("Building FAISS Vector Database...")
db = FAISS.from_documents(chunks, embeddings)

# Ensure the directory exists
os.makedirs("vector_db", exist_ok=True)
db.save_local("vector_db")

# Save chunks for BM25 keyword search
with open("vector_db/docs.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("✅ Vector DB created successfully!")