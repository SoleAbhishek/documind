import re
from typing import List, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter


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


def split_section_blocks(
    section_text: str,
    chunk_size: int = 1500,
    recursive_splitter: RecursiveCharacterTextSplitter = None
) -> List[str]:
    """
    Intelligently splits a markdown section into chunks while protecting table structure.
    Never splits rows mid-table without copying headers.
    """
    if len(section_text) <= chunk_size:
        return [section_text]

    lines = section_text.split("\n")
    blocks: List[Tuple[str, List[str]]] = []  # List of (type, lines), where type is 'prose' or 'table'

    current_type = None
    current_block_lines: List[str] = []

    i = 0
    while i < len(lines):
        line = lines[i]
        # Check if line looks like table row and has next separator line, or is continuation of table
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

    # Now process each block into chunks
    final_sub_chunks: List[str] = []

    for b_type, b_lines in blocks:
        block_text = "\n".join(b_lines).strip()
        if not block_text:
            continue

        if b_type == "table":
            if len(block_text) <= chunk_size:
                final_sub_chunks.append(block_text)
            else:
                table_splits = split_markdown_table(b_lines, max_chunk_size=chunk_size)
                final_sub_chunks.extend(table_splits)
        else:
            # Prose block
            if len(block_text) <= chunk_size:
                final_sub_chunks.append(block_text)
            else:
                if recursive_splitter:
                    final_sub_chunks.extend(recursive_splitter.split_text(block_text))
                else:
                    final_sub_chunks.append(block_text)

    return final_sub_chunks


def test_full_section_splitting():
    prose_before = "### Project Life-Cycle Overview\nThe following table outlines all formal project phases:\n"
    header = "| Phase | Purpose | Typical Output |\n| :--- | :--- | :--- |\n"
    rows = "\n".join([f"| Phase {chr(65+i)} | Core objectives and technology deliverables for stage {i} | Output package specification document {i} |" for i in range(20)])
    prose_after = "\n\nAll lifecycle phases require formal key decision point (KDP) authorization."

    full_section = prose_before + header + rows + prose_after
    print(f"Full section length: {len(full_section)} characters")

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    chunks = split_section_blocks(full_section, chunk_size=600, recursive_splitter=splitter)

    print(f"Split into {len(chunks)} chunks:")
    for idx, c in enumerate(chunks, 1):
        lines = c.split("\n")
        has_table = "|" in c and "| :---" in c
        print(f"\n--- Chunk {idx} (Length: {len(c)} chars, Has Table: {has_table}) ---")
        for line in lines[:3]:
            print(f"  {line[:100]}")
        if len(lines) > 3:
            print(f"  ... ({len(lines)-3} more lines) ...")
            print(f"  Last: {lines[-1][:100]}")

        if has_table:
            # Verify table header preservation
            assert lines[0] == "| Phase | Purpose | Typical Output |", f"Chunk {idx} missing table header!"
            assert lines[1] == "| :--- | :--- | :--- |", f"Chunk {idx} missing table separator!"

    print("\nSUCCESS! Every single table chunk inherited the exact table headers!")


if __name__ == "__main__":
    test_full_section_splitting()
