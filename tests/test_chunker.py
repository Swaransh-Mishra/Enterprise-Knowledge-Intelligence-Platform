from app.ingestion import DocumentIngestion
from app.chunker import DocumentChunker

ingestion = DocumentIngestion()

documents = ingestion.ingest_folder(
    "data/sample_documents"
)

chunker = DocumentChunker()

chunks = chunker.split_documents(documents)

print("=" * 80)
print(f"Total Documents : {len(documents)}")
print(f"Total Chunks    : {len(chunks)}")

first = chunks[0]

print()
print("Chunk Metadata")
print("-" * 40)

print(f"Chunk ID      : {first['chunk_id']}")
print(f"Document      : {first['document_name']}")
print(f"Type          : {first['document_type']}")
print(f"Chunk Number  : {first['chunk_number']}")
print(f"Total Chunks  : {first['total_chunks']}")
print(f"Characters    : {first['characters']}")
print(f"Created At    : {first['created_at']}")

print()
print("Preview")
print("-" * 40)

print(first["text"][:500])