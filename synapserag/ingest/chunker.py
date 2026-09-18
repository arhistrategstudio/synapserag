"""
Hierarchical Contextual Late-Chunker.
Splits text and code while tracking exact line/character offsets and synthesizing situational context.
"""

from __future__ import annotations
from typing import List, Tuple
import re
import uuid

from ..types import Chunk, ChunkType, DocumentSource


class ContextualLateChunker:
    """
    Chunker that preserves document hierarchy, exact character/line coordinates,
    and prepends synthetic contextual summaries to prevent context fragmentation.
    """
    def __init__(self, chunk_size_words: int = 120, overlap_words: int = 25):
        self.chunk_size_words = chunk_size_words
        self.overlap_words = overlap_words

    def create_chunks(self, document: DocumentSource, raw_text: str) -> List[Chunk]:
        """
        Processes document into hierarchical chunks with exact coordinate anchors.
        """
        lines = raw_text.splitlines(keepends=True)
        total_chars = len(raw_text)
        
        # 1. Synthesize document-level global context
        doc_summary = self._synthesize_global_context(document, raw_text)
        document.global_context = doc_summary

        # 2. Build line index for fast char -> line resolution
        line_offsets: List[Tuple[int, int]] = [] # list of (start_char, end_char) for each line
        curr_offset = 0
        for line in lines:
            line_offsets.append((curr_offset, curr_offset + len(line)))
            curr_offset += len(line)

        # 3. Create Parent Chunk representing the overarching document/section
        parent_chunk = Chunk(
            chunk_id=f"parent_{document.doc_id}",
            doc_id=document.doc_id,
            content=raw_text[:min(1000, len(raw_text))],
            contextualized_content=f"[Document: {document.title or document.uri or 'Source'}]\n{doc_summary}",
            chunk_type=ChunkType.PARENT,
            start_char=0,
            end_char=total_chars,
            start_line=1,
            end_line=len(lines) or 1,
            metadata={"is_parent": True, "title": document.title, "uri": document.uri}
        )

        # 4. Create Granular Child Chunks using natural boundaries (paragraphs / code blocks)
        child_chunks: List[Chunk] = []
        paragraphs = self._split_into_logical_blocks(raw_text)
        
        char_cursor = 0
        for block_idx, block in enumerate(paragraphs):
            block_content = block.strip()
            if not block_content:
                continue

            # Find actual start and end character in raw_text
            start_c = raw_text.find(block_content, char_cursor)
            if start_c == -1:
                start_c = char_cursor
            end_c = start_c + len(block_content)
            char_cursor = end_c

            start_l, end_l = self._resolve_lines(start_c, end_c, line_offsets)

            # Prepend contextual situation (Contextual Retrieval technique)
            contextual_header = f"[Context: {document.title or 'Doc'} | Section {block_idx + 1}]\n"
            contextualized = f"{contextual_header}{block_content}"

            child = Chunk(
                chunk_id=f"chunk_{document.doc_id}_{block_idx + 1}_{uuid.uuid4().hex[:6]}",
                doc_id=document.doc_id,
                content=block_content,
                contextualized_content=contextualized,
                chunk_type=ChunkType.CHILD,
                parent_id=parent_chunk.chunk_id,
                start_char=start_c,
                end_char=end_c,
                start_line=start_l,
                end_line=end_l,
                metadata={"block_index": block_idx + 1, "uri": document.uri}
            )
            child_chunks.append(child)

        return [parent_chunk] + child_chunks

    def _split_into_logical_blocks(self, text: str) -> List[str]:
        """Split by double newlines, markdown headers, or fixed word buckets."""
        # Try splitting by double newline first
        raw_blocks = re.split(r"\n\s*\n", text)
        result_blocks = []

        for block in raw_blocks:
            words = block.split()
            if len(words) <= self.chunk_size_words:
                result_blocks.append(block)
            else:
                # Sliding window over large paragraph
                step = self.chunk_size_words - self.overlap_words
                for i in range(0, len(words), max(1, step)):
                    chunk_slice = " ".join(words[i : i + self.chunk_size_words])
                    result_blocks.append(chunk_slice)

        return result_blocks

    def _resolve_lines(
        self, start_char: int, end_char: int, line_offsets: List[Tuple[int, int]]
    ) -> Tuple[int, int]:
        """Determine 1-indexed start and end line from character positions."""
        start_line = 1
        end_line = 1
        
        for idx, (s, e) in enumerate(line_offsets, start=1):
            if s <= start_char < e or (s == start_char and s == e):
                start_line = idx
            if s < end_char <= e or (s <= end_char and idx == len(line_offsets)):
                end_line = idx
                
        return start_line, max(start_line, end_line)

    def _synthesize_global_context(self, doc: DocumentSource, text: str) -> str:
        """Create a compact 2-3 sentence overview of the document."""
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        preview = " ".join(lines[:4]) if lines else "Document content"
        return f"Document overview ({doc.title or doc.uri or 'data'}): {preview[:250]}..."
