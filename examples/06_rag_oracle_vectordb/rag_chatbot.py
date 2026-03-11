"""RAG Chatbot -- Question answering over documents using Oracle Vector Search.

Prerequisites:
    docker compose up -d  # Start Oracle 26ai

Usage:
    export OCI_COMPARTMENT_ID=ocid1.compartment.oc1..xxx
    python rag_chatbot.py

Requirements:
    pip install oci-genai-service oracledb
"""

from oci_genai_service import GenAIClient
from oci_genai_service.vectordb import OracleVectorStore
from oci_genai_service.rag import RAGPipeline

client = GenAIClient()
store = OracleVectorStore(
    dsn="localhost:1521/FREEPDB1",
    user="genai",
    password="genai",
    table_name="rag_demo",
)

pipeline = RAGPipeline(client=client, vector_store=store)

# Ingest a document
chunks = pipeline.ingest("sample_doc.txt")
print(f"Ingested {chunks} chunks")

# Ask a question
answer = pipeline.query("What are the main topics in the document?")
print(f"\nAnswer: {answer.text}")
print(f"\nSources: {len(answer.sources)}")
for s in answer.sources:
    print(f"  - Score: {s['score']:.3f} | {s['chunk'][:60]}...")
