from app.ingestion import DocumentIngestion

pipeline = DocumentIngestion()

documents = pipeline.ingest_folder(
    "data/sample_documents"
)

print("=" * 80)
print(f"Documents Loaded : {len(documents)}")
print("=" * 80)

for doc in documents:

    print(f"\nFile : {doc['filename']}")
    print(f"Type : {doc['extension']}")
    print(f"Size : {doc['size_kb']} KB")

    print("\nPreview:")
    print(doc["text"][:300])

    print("-" * 80)