"""
Hierarchical Contextual Late-Chunker.
Splits text and code while tracking exact line/character offsets and synthesizing situational context.
"""

from __future__ import annotations
from bisect import bisect_right
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
        # Lines are counted on "\n" only (what editors and `file:line` references use);
        # str.splitlines() would also break on form feeds / \x1c-\x1e / \u2028, which are
        # common in PDF-extracted text and would shift every line number after them.
        lines = raw_text.split("\n") if raw_text else []
        total_chars = len(raw_text)
        
        # 1. Synthesize document-level global context
        doc_summary = self._synthesize_global_context(document, raw_text)
        document.global_context = doc_summary

        # 2. Build line index for fast char -> line resolution
        line_offsets: List[Tuple[int, int]] = [] # list of (start_char, end_char) for each line
        curr_offset = 0
        for line in lines:
            line_offsets.append((curr_offset, curr_offset + len(line) + 1))
            curr_offset += len(line) + 1

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

        # 4. Create Granular Child Chunks using natural boundaries (paragraphs / code blocks).
        # Each block is an exact (start_char, end_char) span of raw_text, so content,
        # character offsets and line numbers always agree with the original document.
        child_chunks: List[Chunk] = []
        line_starts = [s for s, _ in line_offsets]

        for block_idx, (start_c, end_c) in enumerate(self._split_into_logical_spans(raw_text)):
            block_content = raw_text[start_c:end_c]
            start_l, end_l = self._resolve_lines(start_c, end_c, line_starts)

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
        """Backward-compatible helper: the text of each logical block."""
        return [text[s:e] for s, e in self._split_into_logical_spans(text)]

    def _split_into_logical_spans(self, text: str) -> List[Tuple[int, int]]:
        """
        Split by blank lines into paragraphs; paragraphs longer than ``chunk_size_words``
        are cut into overlapping word windows. Returns exact (start_char, end_char) spans
        into ``text`` (surrounding whitespace trimmed) instead of re-joined strings, which
        previously broke offset/line resolution for every window after the first.
        """
        spans: List[Tuple[int, int]] = []
        para_start = 0
        separators = list(re.finditer(r"\n\s*\n", text)) + [None]
        for sep in separators:
            para_end = sep.start() if sep else len(text)
            words = [(m.start() + para_start, m.end() + para_start)
                     for m in re.finditer(r"\S+", text[para_start:para_end])]
            if words:
                if len(words) <= self.chunk_size_words:
                    spans.append((words[0][0], words[-1][1]))
                else:
                    step = max(1, self.chunk_size_words - self.overlap_words)
                    for i in range(0, len(words), step):
                        window = words[i:i + self.chunk_size_words]
                        spans.append((window[0][0], window[-1][1]))
                        if i + self.chunk_size_words >= len(words):
                            break
            if sep:
                para_start = sep.end()
        return spans

    def _resolve_lines(
        self, start_char: int, end_char: int, line_starts: List[int]
    ) -> Tuple[int, int]:
        """Determine 1-indexed start and end line from character positions (binary search)."""
        if not line_starts:
            return 1, 1
        start_line = max(1, bisect_right(line_starts, start_char))
        end_line = max(1, bisect_right(line_starts, max(start_char, end_char - 1)))
        return start_line, max(start_line, end_line)

    def _synthesize_global_context(self, doc: DocumentSource, text: str) -> str:
        """Create a compact 2-3 sentence overview of the document."""
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        preview = " ".join(lines[:4]) if lines else "Document content"
        return f"Document overview ({doc.title or doc.uri or 'data'}): {preview[:250]}..."
