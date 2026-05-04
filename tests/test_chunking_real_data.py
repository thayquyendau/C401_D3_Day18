"""
Test chunking strategies với dữ liệu thật
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.m1_chunking import (
    chunk_basic, chunk_semantic, chunk_hierarchical,
    chunk_structure_aware, compare_strategies
)

def test_with_real_data():
    # Load real documents
    docs = []

    # BCTC - Tax form
    with open("data/BCTC.md", encoding="utf-8") as f:
        docs.append({
            "text": f.read(),
            "metadata": {"source": "BCTC.md", "type": "tax_form"}
        })

    # Nghị định - Government decree
    with open("data/Nghi_dinh_so_13-2023_ve_bao_ve_du_lieu_ca_nhan_508ee.md", encoding="utf-8") as f:
        docs.append({
            "text": f.read(),
            "metadata": {"source": "Nghi_dinh.md", "type": "legal_doc"}
        })

    print("=" * 80)
    print("TESTING CHUNKING STRATEGIES WITH REAL VIETNAMESE DOCUMENTS")
    print("=" * 80)

    # Compare all strategies
    print("\n📊 COMPARISON OF ALL STRATEGIES\n")
    results = compare_strategies(docs)

    print(f"{'Strategy':<20} {'Chunks':<10} {'Avg Len':<10} {'Min':<10} {'Max':<10}")
    print("-" * 60)

    for strategy, stats in results.items():
        if strategy == "hierarchical":
            print(f"{strategy:<20} {stats['num_children']:<10} {stats['avg_length']:<10} {stats['min_length']:<10} {stats['max_length']:<10}")
            print(f"  └─ parents: {stats['num_parents']}")
        else:
            print(f"{strategy:<20} {stats['num_chunks']:<10} {stats['avg_length']:<10} {stats['min_length']:<10} {stats['max_length']:<10}")

    # Test with first document (BCTC)
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS: BCTC.md (Tax Form)")
    print("=" * 80)

    text = docs[0]["text"]

    # Basic chunking
    basic_chunks = chunk_basic(text, chunk_size=500)
    print(f"\n1️⃣  BASIC CHUNKING: {len(basic_chunks)} chunks")
    print("\nSample chunk 1:")
    print(f"  Length: {len(basic_chunks[0].text)} chars")
    print(f"  Preview: {basic_chunks[0].text[:200]}...")

    # Semantic chunking
    semantic_chunks = chunk_semantic(text, threshold=0.7)
    print(f"\n2️⃣  SEMANTIC CHUNKING: {len(semantic_chunks)} chunks")
    print("\nSample chunk 1:")
    print(f"  Length: {len(semantic_chunks[0].text)} chars")
    print(f"  Preview: {semantic_chunks[0].text[:200]}...")

    # Hierarchical chunking
    parents, children = chunk_hierarchical(text, parent_size=1500, child_size=300)
    print(f"\n3️⃣  HIERARCHICAL CHUNKING: {len(parents)} parents, {len(children)} children")
    print("\nSample parent:")
    print(f"  Length: {len(parents[0].text)} chars")
    print(f"  Preview: {parents[0].text[:200]}...")
    print("\nSample child (linked to parent):")
    print(f"  Parent ID: {children[0].parent_id}")
    print(f"  Length: {len(children[0].text)} chars")
    print(f"  Preview: {children[0].text[:150]}...")

    # Structure-aware chunking
    structure_chunks = chunk_structure_aware(text)
    print(f"\n4️⃣  STRUCTURE-AWARE CHUNKING: {len(structure_chunks)} chunks")
    print("\nSample chunks with headers:")
    for i, chunk in enumerate(structure_chunks[:3]):
        section = chunk.metadata.get("section", "No header")
        print(f"\n  Chunk {i+1}: Section '{section}'")
        print(f"    Length: {len(chunk.text)} chars")
        print(f"    Preview: {chunk.text[:150]}...")

    # Test with second document (Nghị định)
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS: Nghị định (Legal Document)")
    print("=" * 80)

    text2 = docs[1]["text"]

    # Structure-aware is best for legal docs
    structure_chunks2 = chunk_structure_aware(text2)
    print(f"\n📋 STRUCTURE-AWARE CHUNKING: {len(structure_chunks2)} chunks")
    print("\nFirst 5 sections:")
    for i, chunk in enumerate(structure_chunks2[:5]):
        section = chunk.metadata.get("section", "No header")
        level = chunk.metadata.get("level", 0)
        print(f"  {'  ' * (level-1)}└─ {section} ({len(chunk.text)} chars)")

    print("\n" + "=" * 80)
    print("✅ TESTING COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    test_with_real_data()
