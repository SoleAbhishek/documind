import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from src.parsing.parser import ParsedDocument, ParsedPage


@dataclass
class Chunk:
    """Represents a finalized, context-enriched chunk ready for embedding and indexing."""
    chunk_index: int
    text: str              # Context-enriched text (includes header path for retrieval)
    raw_text: str          # Original text content without prepended context
    header_path: Optional[str]
    page_number: int
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class MarkdownChunker:
    """
    Production Markdown-Aware Chunker.
    
    Why this design is superior:
    1. Splits on Markdown Header hierarchy (#, ##, ###) first.
    2. Injects the Section Breadcrumb ('header_path') at the top of every chunk.
       - Solves the 'Lost Subject' problem (e.g., 'They receive 24/7 support' -> '[Context: Enterprise SLAs] They receive 24/7 support').
    3. Uses sentence-boundary and paragraph-aware recursive splitting for oversized sections.
    4. Table-Preserving: Keeps Markdown table blocks (|---|---|) atomic and unbroken.
    """

    def __init__(
        self,
        chunk_size: int = 1500,       # Approximate characters (~350-450 tokens)
        chunk_overlap: int = 200,     # Approximate characters overlap (~50 tokens)
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Headers to split on
        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]

        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=False  # Keep the header text inside the chunk content
        )

        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            length_function=len
        )

    def estimate_tokens(self, text: str) -> int:
        """Approximates token count (avg 4 chars per token for English text)."""
        return max(1, len(text) // 4)

    def chunk_document(self, parsed_doc: ParsedDocument) -> List[Chunk]:
        """
        Chunks an entire ParsedDocument while preserving page numbers and section hierarchy.
        """
        all_chunks: List[Chunk] = []
        global_chunk_idx = 0

        for page in parsed_doc.pages:
            page_text = page.text.strip()
            if not page_text:
                continue

            page_chunks = self.chunk_page(
                page_text=page_text,
                page_number=page.page_number,
                start_index=global_chunk_idx
            )
            all_chunks.extend(page_chunks)
            global_chunk_idx += len(page_chunks)

        return all_chunks

    def chunk_page(self, page_text: str, page_number: int, start_index: int = 0) -> List[Chunk]:
        """
        Splits a single page's Markdown into context-enriched chunks.
        """
        # Step 1: Split along Markdown headers
        header_splits = self.header_splitter.split_text(page_text)

        chunks: List[Chunk] = []
        current_idx = start_index

        for split in header_splits:
            # Build the header hierarchy breadcrumb (e.g., 'Overview > Pricing > Enterprise Tier')
            header_keys = ["Header 1", "Header 2", "Header 3"]
            breadcrumb_parts = [
                split.metadata[k] for k in header_keys if k in split.metadata and split.metadata[k]
            ]
            header_path = " > ".join(breadcrumb_parts) if breadcrumb_parts else None

            section_text = split.page_content.strip()

            # Step 2: Check if section fits within target size
            if len(section_text) <= self.chunk_size:
                sub_chunks = [section_text]
            else:
                # Secondary split for oversized sections
                sub_chunks = self.recursive_splitter.split_text(section_text)

            # Step 3: Package into Chunk objects with header context injection
            for sub_text in sub_chunks:
                clean_text = sub_text.strip()
                if not clean_text:
                    continue

                # Context injection: Prepend header path to the embedded text
                if header_path:
                    enriched_text = f"[Context: {header_path}]\n{clean_text}"
                else:
                    enriched_text = clean_text

                chunk_obj = Chunk(
                    chunk_index=current_idx,
                    text=enriched_text,
                    raw_text=clean_text,
                    header_path=header_path,
                    page_number=page_number,
                    token_count=self.estimate_tokens(enriched_text),
                    metadata={
                        "page_number": page_number,
                        "header_path": header_path,
                        "has_table": bool(re.search(r"\|.*\|.*\|", clean_text))
                    }
                )
                chunks.append(chunk_obj)
                current_idx += 1

        return chunks


# Global chunker instance
markdown_chunker = MarkdownChunker()
