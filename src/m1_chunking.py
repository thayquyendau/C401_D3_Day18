"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: Optional[str] = None


def load_documents(data_dir: str = DATA_DIR) -> List[Dict]:
    """Load all markdown/text files from data/. (Đã implement sẵn)"""
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})
    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: Optional[dict] = None) -> List[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: Optional[dict] = None) -> List[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.

    Args:
        text: Input text.
        threshold: Cosine similarity threshold. Dưới threshold → tách chunk mới.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects grouped by semantic similarity.
    """
    metadata = metadata or {}

    # Split text into sentences
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n\n', text) if s.strip()]
    if not sentences:
        return []

    # Encode sentences
    from sentence_transformers import SentenceTransformer
    import numpy as np

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(sentences)

    # Group sentences by similarity
    chunks = []
    current_group = [sentences[0]]

    for i in range(1, len(sentences)):
        # Cosine similarity between consecutive sentences
        sim = np.dot(embeddings[i-1], embeddings[i]) / (
            np.linalg.norm(embeddings[i-1]) * np.linalg.norm(embeddings[i])
        )

        if sim < threshold:
            # Create new chunk
            chunk_text = " ".join(current_group)
            chunks.append(Chunk(
                text=chunk_text,
                metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"}
            ))
            current_group = []

        current_group.append(sentences[i])

    # Add last group
    if current_group:
        chunk_text = " ".join(current_group)
        chunks.append(Chunk(
            text=chunk_text,
            metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"}
        ))

    return chunks


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: Optional[dict] = None) -> Tuple[List[Chunk], List[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Args:
        text: Input text.
        parent_size: Chars per parent chunk.
        child_size: Chars per child chunk.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    parents = []
    children = []

    # Split text into paragraphs
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Create parent chunks
    current_parent = ""
    p_index = 0

    for para in paragraphs:
        if len(current_parent) + len(para) > parent_size and current_parent:
            pid = f"parent_{p_index}"
            parents.append(Chunk(
                text=current_parent.strip(),
                metadata={**metadata, "chunk_type": "parent", "parent_id": pid}
            ))

            # Split parent into children
            parent_text = current_parent.strip()
            for i in range(0, len(parent_text), child_size):
                child_text = parent_text[i:i + child_size]
                if child_text.strip():
                    children.append(Chunk(
                        text=child_text.strip(),
                        metadata={**metadata, "chunk_type": "child"},
                        parent_id=pid
                    ))

            current_parent = ""
            p_index += 1

        current_parent += para + "\n\n"

    # Handle last parent
    if current_parent.strip():
        pid = f"parent_{p_index}"
        parents.append(Chunk(
            text=current_parent.strip(),
            metadata={**metadata, "chunk_type": "parent", "parent_id": pid}
        ))

        parent_text = current_parent.strip()
        for i in range(0, len(parent_text), child_size):
            child_text = parent_text[i:i + child_size]
            if child_text.strip():
                children.append(Chunk(
                    text=child_text.strip(),
                    metadata={**metadata, "chunk_type": "child"},
                    parent_id=pid
                ))

    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: Optional[dict] = None) -> List[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.

    Args:
        text: Markdown text.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects, mỗi chunk = 1 section (header + content).
    """
    metadata = metadata or {}

    # Split by markdown headers (h1-h3)
    sections = re.split(r'(^#{1,3}\s+.+$)', text, flags=re.MULTILINE)

    chunks = []
    current_header = ""
    current_content = ""

    for part in sections:
        if re.match(r'^#{1,3}\s+', part):
            # Save previous section
            if current_content.strip():
                chunk_text = f"{current_header}\n{current_content}".strip() if current_header else current_content.strip()
                level = len(re.match(r'^(#+)', current_header).group(1)) if current_header else 0
                chunks.append(Chunk(
                    text=chunk_text,
                    metadata={**metadata, "section": current_header.strip('#').strip(), "level": level, "strategy": "structure"}
                ))

            current_header = part.strip()
            current_content = ""
        else:
            current_content += part

    # Add last section
    if current_header or current_content.strip():
        chunk_text = f"{current_header}\n{current_content}".strip() if current_header else current_content.strip()
        level = len(re.match(r'^(#+)', current_header).group(1)) if current_header else 0
        chunks.append(Chunk(
            text=chunk_text,
            metadata={**metadata, "section": current_header.strip('#').strip() if current_header else "", "level": level, "strategy": "structure"}
        ))

    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: List[dict]) -> dict:
    """
    Run all strategies on documents and compare.

    Returns:
        {"basic": {...}, "semantic": {...}, "hierarchical": {...}, "structure": {...}}
    """
    results = {}

    # Collect all chunks from all documents for each strategy
    all_basic = []
    all_semantic = []
    all_hierarchical_parents = []
    all_hierarchical_children = []
    all_structure = []

    for doc in documents:
        text = doc["text"]
        meta = doc.get("metadata", {})

        # Run each strategy
        all_basic.extend(chunk_basic(text, metadata=meta))
        all_semantic.extend(chunk_semantic(text, metadata=meta))
        parents, children = chunk_hierarchical(text, metadata=meta)
        all_hierarchical_parents.extend(parents)
        all_hierarchical_children.extend(children)
        all_structure.extend(chunk_structure_aware(text, metadata=meta))

    # Calculate stats for each strategy
    def calc_stats(chunks):
        if not chunks:
            return {"num_chunks": 0, "avg_length": 0, "min_length": 0, "max_length": 0}
        lengths = [len(c.text) for c in chunks]
        return {
            "num_chunks": len(chunks),
            "avg_length": int(sum(lengths) / len(lengths)),
            "min_length": min(lengths),
            "max_length": max(lengths)
        }

    results["basic"] = calc_stats(all_basic)
    results["semantic"] = calc_stats(all_semantic)
    results["hierarchical"] = {
        "num_parents": len(all_hierarchical_parents),
        "num_children": len(all_hierarchical_children),
        **calc_stats(all_hierarchical_children)
    }
    results["structure"] = calc_stats(all_structure)

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
