import json
import logging
from pathlib import Path
from typing import Any

from docling.document_converter import DocumentConverter

from app.core.config import CLEAR_DOCS_DIR, RAW_DOCS_DIR

logger = logging.getLogger(__name__)


class Chunker:
    def __init__(self, raw_docs_dir: str = None, processed_docs_dir: str = None):
        self.raw_docs_dir = Path(raw_docs_dir or RAW_DOCS_DIR)
        self.clear_docs_dir = Path(processed_docs_dir or CLEAR_DOCS_DIR)
        self.converter = DocumentConverter()

        # Create directories if they don't exist
        self.raw_docs_dir.mkdir(exist_ok=True)
        self.clear_docs_dir.mkdir(exist_ok=True)

    def _split_paragraph_recursive(
        self, text: str, max_size: int, overlap: int
    ) -> list[str]:
        """
        Recursively splits text into chunks if it exceeds max_size.
        """
        chunks = []

        # If text is already smaller or equal to max size, return as is
        if len(text) <= max_size:
            chunks.append(text)
            return chunks

        # Find optimal split point - try to split by sentence or word
        split_point = self._find_optimal_split_point(text, max_size)

        if split_point == -1:
            # If no good split point found, split simply by max_size
            split_point = max_size

        # First chunk
        first_chunk = text[:split_point].strip()
        if first_chunk:
            chunks.append(first_chunk)

        # Second chunk with overlap
        remaining_text = (
            text[split_point - overlap :]
            if split_point > overlap
            else text[split_point:]
        )
        if remaining_text:
            # Recursively process remaining text
            chunks.extend(
                self._split_paragraph_recursive(remaining_text, max_size, overlap)
            )

        return chunks

    def _find_optimal_split_point(self, text: str, max_size: int) -> int:
        """
        Finds optimal point for text splitting.
        Prefers sentence endings, then commas, then word spaces.
        """
        # Look for sentence endings within max_size
        sentence_endings = [".", "!", "?", "。", "！", "？"]
        for i in range(min(max_size, len(text)) - 1, max(0, max_size - 100), -1):
            if text[i] in sentence_endings and (
                i + 1 >= len(text) or text[i + 1] in [" ", "\n", '"', "'"]
            ):
                return i + 1

        # Look for commas
        for i in range(min(max_size, len(text)) - 1, max(0, max_size - 50), -1):
            if text[i] == "," and text[i + 1] == " ":
                return i + 1

        # Look for spaces between words
        for i in range(min(max_size, len(text)) - 1, max(0, max_size - 30), -1):
            if text[i] == " ":
                return i + 1

        return -1  # No good split point found

    def _extract_text_from_items(self, items) -> list[dict[str, Any]]:
        """
        Extracts text from Docling document items.
        """
        paragraphs = []

        for item in items:
            # Get text representation of the element
            text_content = item.text if hasattr(item, "text") else str(item)

            if text_content and text_content.strip():
                # Determine element type for title
                element_type = type(item).__name__

                paragraphs.append(
                    {
                        "text": text_content.strip(),
                        "title": f"{element_type}_{len(paragraphs) + 1}",
                        "element_type": element_type,
                        "order": len(paragraphs),
                    }
                )

        return paragraphs

    def process_document(
        self, filename: str, max_chunk_size: int
    ) -> list[dict[str, Any]]:
        """
        Main document processing method.
        """
        file_path = self.raw_docs_dir / filename

        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} not found")

        logger.info(f"Processing file: {filename}")

        # Convert document
        result = self.converter.convert(str(file_path))

        chunks = []
        chunk_id = 0
        overlap_size = int(max_chunk_size * 0.1)  # 10% overlap

        # Extract text from document elements
        if hasattr(result.document, "texts"):
            paragraphs = self._extract_text_from_items(result.document.texts)
        else:
            # Alternative method: get all text as string
            full_text = str(result.document)
            paragraphs = [
                {
                    "text": full_text,
                    "title": "Full_Document",
                    "element_type": "Document",
                    "order": 0,
                }
            ]

        logger.info(f"Found {len(paragraphs)} text elements for processing")

        for para_data in paragraphs:
            paragraph = para_data["text"]
            section_title = para_data["title"]
            element_type = para_data["element_type"]

            if not paragraph.strip():
                continue

            # Process each paragraph
            if len(paragraph) <= max_chunk_size:
                # Paragraph fits in one chunk
                chunk_data = {
                    "chunk_id": chunk_id,
                    "section_title": section_title,
                    "element_type": element_type,
                    "text": paragraph,
                    "chunk_size": len(paragraph),
                    "is_split": False,
                    "original_paragraph_size": len(paragraph),
                }
                chunks.append(chunk_data)
                chunk_id += 1
            else:
                # Paragraph needs to be split
                paragraph_chunks = self._split_paragraph_recursive(
                    paragraph, max_chunk_size, overlap_size
                )

                for i, chunk_text in enumerate(paragraph_chunks):
                    chunk_data = {
                        "chunk_id": chunk_id,
                        "section_title": section_title,
                        "element_type": element_type,
                        "text": chunk_text,
                        "chunk_size": len(chunk_text),
                        "is_split": True,
                        "split_part": f"{i+1}/{len(paragraph_chunks)}",
                        "original_paragraph_size": len(paragraph),
                    }
                    chunks.append(chunk_data)
                    chunk_id += 1

        # Save result
        output_filename = f"{Path(filename).stem}_chunks.json"
        output_path = self.clear_docs_dir / output_filename

        output_data = {
            "source_file": filename,
            "max_chunk_size": max_chunk_size,
            "overlap_size": overlap_size,
            "total_chunks": len(chunks),
            "chunks": chunks,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Processing completed. Created {len(chunks)} chunks.")
        logger.info(f"Result saved to: {output_path}")

        return chunks
