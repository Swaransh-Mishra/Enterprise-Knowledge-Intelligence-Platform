"""
Document Ingestion Pipeline.

Responsible for:
1. Scanning folders
2. Finding supported documents
3. Loading document contents
4. Returning structured document objects
"""

from pathlib import Path
from app.text_cleaner import TextCleaner
from app.document_loader import DocumentLoader


class DocumentIngestion:

    SUPPORTED_EXTENSIONS = {
        ".pdf",
        ".docx",
        ".txt",
        ".md",
        ".csv",
        ".xlsx",
        ".pptx",
        ".html",
        ".json",
    }

    def __init__(self):

        self.loader = DocumentLoader()
        self.cleaner = TextCleaner()

    def ingest_folder(self, folder_path: str):

        folder = Path(folder_path)

        if not folder.exists():
            raise FileNotFoundError(f"{folder_path} does not exist.")

        documents = []

        for file in folder.iterdir():

            if not file.is_file():
                continue

            if file.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                continue

            try:

                raw_text = self.loader.load_document(str(file))

                text = self.cleaner.clean(raw_text)

                documents.append(
                    {
                        "filename": file.name,
                        "extension": file.suffix.lower(),
                        "size_kb": round(file.stat().st_size / 1024, 2),
                        "text": text,
                    }
                )

            except Exception as e:

                print(f"Skipped {file.name}: {e}")

        return documents