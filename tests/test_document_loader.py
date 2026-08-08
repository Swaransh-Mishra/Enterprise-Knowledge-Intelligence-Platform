from app.document_loader import DocumentLoader

loader = DocumentLoader()

text = loader.load_document(
    "data/sample_documents/sample.pdf"
)

print("=" * 80)
print(text[:1500])
print("=" * 80)