"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import json
import os
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


def _can_use_openai() -> bool:
    """Return True only when the API key looks usable."""
    return bool(OPENAI_API_KEY and not OPENAI_API_KEY.startswith("sk-..."))


def _call_openai(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 200,
    json_response: bool = False,
) -> str | None:
    """Best-effort OpenAI call; callers provide deterministic fallbacks."""
    if not _can_use_openai():
        return None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_API_KEY)
        kwargs = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }
        if json_response:
            kwargs["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content.strip()
    except Exception:
        return None


def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?。！？])\s+|\n+", text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def _looks_vietnamese(text: str) -> bool:
    return bool(re.search(r"[ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩị"
                          r"óòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]", text.lower()))


def _infer_category(text: str) -> str:
    lowered = text.lower()
    if any(word in lowered for word in ["nghỉ phép", "nhân viên", "thâm niên", "hợp đồng", "lương"]):
        return "hr"
    if any(word in lowered for word in ["mật khẩu", "vpn", "tài khoản", "bảo mật", "hệ thống"]):
        return "it"
    if any(word in lowered for word in ["doanh thu", "chi phí", "báo cáo tài chính", "thuế"]):
        return "finance"
    if any(word in lowered for word in ["nghị định", "điều", "quy định", "chính sách"]):
        return "policy"
    return "general"


def _infer_topic(text: str) -> str:
    lowered = text.lower()
    topic_keywords = [
        "nghỉ phép", "bảo vệ dữ liệu cá nhân", "mật khẩu", "vpn",
        "báo cáo tài chính", "nhân viên", "chính sách", "quy định",
    ]
    for keyword in topic_keywords:
        if keyword in lowered:
            return keyword

    first_sentence = _split_sentences(text)
    if not first_sentence:
        return "unknown"
    words = re.findall(r"\w+", first_sentence[0], flags=re.UNICODE)
    return " ".join(words[:6]).lower() if words else "unknown"


def _extract_entities_heuristic(text: str) -> list[str]:
    known_entities = [
        "VinUni", "OpenAI", "Qdrant", "RAGAS", "VPN",
        "Nghị định 13", "BCTC",
    ]
    entities = [entity for entity in known_entities if entity.lower() in text.lower()]
    entities.extend(re.findall(r"\b[A-ZĐ][\wĐđ]*(?:\s+[A-ZĐ][\wĐđ]*){0,3}", text))

    seen = set()
    unique_entities = []
    for entity in entities:
        clean = entity.strip()
        if clean and clean.lower() not in seen:
            unique_entities.append(clean)
            seen.add(clean.lower())
    return unique_entities[:8]


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.

    Args:
        text: Raw chunk text.

    Returns:
        Summary string (2-3 câu).
    """
    text = text.strip()
    if not text:
        return ""

    llm_summary = _call_openai(
        "Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt.",
        text,
        max_tokens=150,
    )
    if llm_summary:
        return llm_summary

    sentences = _split_sentences(text)
    if not sentences:
        return text[:300].strip()
    return " ".join(sentences[:2]).strip()


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).

    Args:
        text: Raw chunk text.
        n_questions: Số câu hỏi cần generate.

    Returns:
        List of question strings.
    """
    text = text.strip()
    if not text or n_questions <= 0:
        return []

    llm_questions = _call_openai(
        f"Dựa trên đoạn văn, tạo đúng {n_questions} câu hỏi mà đoạn văn có thể trả lời. "
        "Trả về mỗi câu hỏi trên một dòng, không thêm giải thích.",
        text,
        max_tokens=200,
    )
    if llm_questions:
        parsed = []
        for line in llm_questions.splitlines():
            question = re.sub(r"^\s*[-*]?\s*\d*[\).:-]?\s*", "", line).strip()
            if question:
                if not question.endswith("?"):
                    question += "?"
                parsed.append(question)
        if parsed:
            return parsed[:n_questions]

    topic = _infer_topic(text)
    category = _infer_category(text)
    questions = [
        f"Nội dung này nói về {topic} như thế nào?",
        f"Quy định liên quan đến {topic} là gì?",
        f"Thông tin chính về {topic} trong tài liệu là gì?",
    ]
    if category == "hr":
        questions.insert(0, f"Nhân viên cần biết gì về {topic}?")
    elif category == "it":
        questions.insert(0, f"Yêu cầu IT liên quan đến {topic} là gì?")
    elif category == "finance":
        questions.insert(0, f"Số liệu tài chính liên quan đến {topic} là gì?")
    return questions[:n_questions]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).

    Args:
        text: Raw chunk text.
        document_title: Tên document gốc.

    Returns:
        Text với context prepended.
    """
    text = text.strip()
    if not text:
        return ""

    prompt = f"Tài liệu: {document_title or 'Không rõ'}\n\nĐoạn văn:\n{text}"
    context = _call_openai(
        "Viết 1 câu ngắn mô tả đoạn văn này nằm ở đâu trong tài liệu và nói về chủ đề gì. "
        "Chỉ trả về 1 câu.",
        prompt,
        max_tokens=80,
    )
    if not context:
        source = f" trong tài liệu {document_title}" if document_title else ""
        context = f"Đoạn trích này{source} nói về {_infer_topic(text)}."
    return f"{context.strip()}\n\n{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.

    Args:
        text: Raw chunk text.

    Returns:
        Dict with extracted metadata fields.
    """
    text = text.strip()
    if not text:
        return {
            "topic": "unknown",
            "entities": [],
            "category": "general",
            "language": "unknown",
        }

    llm_metadata = _call_openai(
        'Trích xuất metadata từ đoạn văn. Trả về JSON đúng schema: '
        '{"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance|general", '
        '"language": "vi|en"}',
        text,
        max_tokens=150,
        json_response=True,
    )
    if llm_metadata:
        try:
            parsed = json.loads(llm_metadata)
            return {
                "topic": str(parsed.get("topic") or _infer_topic(text)),
                "entities": list(parsed.get("entities") or []),
                "category": str(parsed.get("category") or _infer_category(text)),
                "language": str(parsed.get("language") or ("vi" if _looks_vietnamese(text) else "en")),
            }
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    return {
        "topic": _infer_topic(text),
        "entities": _extract_entities_heuristic(text),
        "category": _infer_category(text),
        "language": "vi" if _looks_vietnamese(text) else "en",
    }


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks.

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: List of methods to apply. Default: ["contextual", "hyqa", "metadata"]
                 Options: "summary", "hyqa", "contextual", "metadata", "full"

    Returns:
        List of EnrichedChunk objects.
    """
    if methods is None:
        methods = ["contextual", "hyqa", "metadata"]

    normalized_methods = set(methods)
    use_full = "full" in normalized_methods
    enriched = []

    for chunk in chunks:
        text = str(chunk.get("text", "")).strip()
        metadata = dict(chunk.get("metadata") or {})
        source = str(metadata.get("source", ""))

        summary = summarize_chunk(text) if use_full or "summary" in normalized_methods else ""
        questions = (
            generate_hypothesis_questions(text)
            if use_full or "hyqa" in normalized_methods
            else []
        )
        enriched_text = (
            contextual_prepend(text, source)
            if use_full or "contextual" in normalized_methods
            else text
        )
        auto_meta = extract_metadata(text) if use_full or "metadata" in normalized_methods else {}

        enrichment_blocks = []
        if summary and (use_full or "summary" in normalized_methods):
            enrichment_blocks.append(f"Tóm tắt: {summary}")
        if questions:
            enrichment_blocks.append("Câu hỏi liên quan:\n" + "\n".join(f"- {q}" for q in questions))

        if enrichment_blocks:
            enriched_text = f"{enriched_text}\n\n" + "\n\n".join(enrichment_blocks)

        enriched.append(
            EnrichedChunk(
                original_text=text,
                enriched_text=enriched_text,
                summary=summary,
                hypothesis_questions=questions,
                auto_metadata={**metadata, **auto_meta},
                method="full" if use_full else "+".join(methods),
            )
        )

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
