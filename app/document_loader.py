"""
Multi-Format Document Loader.

Supports:
- PDF
- DOCX
- TXT
- Markdown
- CSV
- Excel
- PowerPoint
- HTML
- JSON
"""

from pathlib import Path
import json

import fitz
import pandas as pd
from bs4 import BeautifulSoup
from docx import Document
from pptx import Presentation


class DocumentLoader:

    def load_document(self, file_path: str) -> str:

        extension = Path(file_path).suffix.lower()

        loaders = {
            ".pdf": self._load_pdf,
            ".docx": self._load_docx,
            ".txt": self._load_txt,
            ".md": self._load_txt,
            ".csv": self._load_csv,
            ".xlsx": self._load_excel,
            ".pptx": self._load_pptx,
            ".html": self._load_html,
            ".json": self._load_json,
        }

        if extension not in loaders:
            raise ValueError(
                f"Unsupported file type: {extension}"
            )

        return loaders[extension](file_path)

    def _load_pdf(self, file_path):

        text = ""

        pdf = fitz.open(file_path)

        for page in pdf:
            text += page.get_text()

        pdf.close()

        return text

    def _load_docx(self, file_path):

        doc = Document(file_path)

        return "\n".join(
            paragraph.text
            for paragraph in doc.paragraphs
        )

    def _load_txt(self, file_path):

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            return file.read()

    def _load_csv(self, file_path):

        df = pd.read_csv(file_path)

        return df.to_string(index=False)

    def _load_excel(self, file_path):

        df = pd.read_excel(file_path)

        return df.to_string(index=False)

    def _load_pptx(self, file_path):

        presentation = Presentation(file_path)

        text = []

        for slide in presentation.slides:

            for shape in slide.shapes:

                if hasattr(shape, "text"):
                    text.append(shape.text)

        return "\n".join(text)

    def _load_html(self, file_path):

        with open(
            file_path,
            encoding="utf-8"
        ) as file:

            soup = BeautifulSoup(
                file.read(),
                "html.parser"
            )

        return soup.get_text(
            separator="\n"
        )

    def _load_json(self, file_path):

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        return json.dumps(
            data,
            indent=2,
            ensure_ascii=False
        )