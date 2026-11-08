import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from src.parsing.parser import ParsedDocument, ParsedPage


def is_table_separator_line(line: str) -> bool:
    """Checks if a line is a markdown table separator (e.g., '| --- | :---: | ---: |')."""
    s = line.strip()
    return bool(re.match(r"^\|(?:\s*:?-+:?\s*\|)+\s*$", s))


def split_markdown_table(table_lines: List[str], max_chunk_size: int = 1500) -> List[str]:
    """
    Splits an oversized Markdown table while guaranteeing that EVERY chunk
    inherits the column headers and alignment separator row.
    """
    if len(table_lines) < 3 or not is_table_separator_line(table_lines[1]):
        return ["\n".join(table_lines)]

    header_line = table_lines[0].strip()
    separator_line = table_lines[1].strip()
    data_rows = table_lines[2:]

    header_block = f"{header_line}\n{separator_line}\n"
    header_len = len(header_block)

    chunks: List[str] = []
    current_rows: List[str] = []
    current_len = header_len

    for row in data_rows:
        row_str = row.strip()
        if not row_str:
            continue
        row_len = len(row_str) + 1  # newline

        if current_rows and (current_len + row_len > max_chunk_size):
            chunks.append(header_block + "\n".join(current_rows))
            current_rows = [row_str]
            current_len = header_len + row_len
        else:
            current_rows.append(row_str)
            current_len += row_len

    if current_rows:
        chunks.append(header_block + "\n".join(current_rows))

    return chunks


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

    def split_section_blocks(self, section_text: str) -> List[str]:
        """
        Intelligently splits a markdown section into chunks while protecting table structure.
        Ensures that split tables always inherit column headers and separator rows in every sub-chunk.
        """
        if len(section_text) <= self.chunk_size:
            return [section_text]

        lines = section_text.split("\n")
        blocks: List[tuple[str, List[str]]] = []

        current_type = None
        current_block_lines: List[str] = []

        i = 0
        while i < len(lines):
            line = lines[i]
            is_pipe_row = "|" in line and line.strip().startswith("|") and line.strip().endswith("|")

            if current_type == "table":
                if is_pipe_row:
                    current_block_lines.append(line)
                    i += 1
                    continue
                else:
                    # End of table
                    blocks.append(("table", current_block_lines))
                    current_type = None
                    current_block_lines = []

            # Check if starting a new markdown table (needs header + separator)
            if is_pipe_row and i + 1 < len(lines) and is_table_separator_line(lines[i + 1]):
                if current_block_lines:
                    blocks.append(("prose", current_block_lines))
                    current_block_lines = []
                current_type = "table"
                current_block_lines = [line, lines[i + 1]]
                i += 2
                continue

            # Regular prose line
            current_block_lines.append(line)
            i += 1

        if current_block_lines:
            blocks.append((current_type or "prose", current_block_lines))

        # Process each block into chunks
        final_sub_chunks: List[str] = []
        for b_type, b_lines in blocks:
            block_text = "\n".join(b_lines).strip()
            if not block_text:
                continue

            if b_type == "table":
                if len(block_text) <= self.chunk_size:
                    final_sub_chunks.append(block_text)
                else:
                    table_splits = split_markdown_table(b_lines, max_chunk_size=self.chunk_size)
                    final_sub_chunks.extend(table_splits)
            else:
                if len(block_text) <= self.chunk_size:
                    final_sub_chunks.append(block_text)
                else:
                    final_sub_chunks.extend(self.recursive_splitter.split_text(block_text))

        return final_sub_chunks

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

            # Step 2: Check if section fits within target size with table-preserving logic
            if len(section_text) <= self.chunk_size:
                sub_chunks = [section_text]
            else:
                # Secondary split with table-header inheritance
                sub_chunks = self.split_section_blocks(section_text)

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

                has_table = bool(re.search(r"^\|(?:\s*:?-+:?\s*\|)+\s*$", clean_text, re.MULTILINE))

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
                        "has_table": has_table
                    }
                )
                chunks.append(chunk_obj)
                current_idx += 1

        return chunks


# Global chunker instance
markdown_chunker = MarkdownChunker()
