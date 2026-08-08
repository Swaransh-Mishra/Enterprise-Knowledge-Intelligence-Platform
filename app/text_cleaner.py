"""
Text Cleaning Module.

Responsible for:
1. Cleaning extracted document text
2. Normalizing whitespace
3. Removing unnecessary blank lines
4. Preparing text for chunking
"""

import re


class TextCleaner:

    def clean(self, text: str) -> str:

        if not text:
            return ""

        text = self._normalize_newlines(text)

        text = self._remove_extra_spaces(text)

        text = self._remove_extra_blank_lines(text)

        text = self._remove_control_characters(text)

        return text.strip()

    def _normalize_newlines(self, text: str) -> str:

        return text.replace("\r\n", "\n").replace("\r", "\n")

    def _remove_extra_spaces(self, text: str) -> str:

        return re.sub(r"[ \t]+", " ", text)

    def _remove_extra_blank_lines(self, text: str) -> str:

        return re.sub(r"\n{3,}", "\n\n", text)

    def _remove_control_characters(self, text: str) -> str:

        return re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", text)