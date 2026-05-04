# Group Report — Lab 18: Production RAG

**Nhóm:** Nhóm D3
**Ngày:** 04/05/2026

## Thành viên & Phân công

| Tên | Module | Hoàn thành | Tests pass |
|-----|--------|-----------|-----------|
| Thành viên 1 | M1: Chunking | ☑ | 8/8 |
| 2A202600144_Nguyễn Doãn Hiếu | M2: Hybrid Search | ☑ | 5/5 |
| Thành viên 3 | M3: Reranking | ☑ | 5/5 |
| Thành viên 4 | M4: Evaluation | ☑ | 4/4 |

## Kết quả RAGAS

| Metric | Naive | Production | Δ |
|--------|-------|-----------|---|
| Faithfulness | 0.7917 | 0.9444 | +0.1527 |
| Answer Relevancy | 0.3967 | 0.8475 | +0.4508 |
| Context Precision | 0.7083 | 0.8240 | +0.1157 |
| Context Recall | 0.6944 | 0.9108 | +0.2164 |

## Key Findings

1. **Biggest improvement:** Answer Relevancy tăng mạnh nhất (+0.4508). Việc sử dụng Hybrid Search và Reranking giúp trả về đúng chunk chứa thông tin, giúp LLM trả lời trúng đích hơn thay vì trả về câu trả lời chung chung hoặc không liên quan.
2. **Biggest challenge:** Tối ưu hóa Context Precision khi có quá nhiều chunk nhiễu lọt vào kết quả search (đặc biệt với các câu hỏi về tài liệu cụ thể như Tờ khai thuế).
3. **Surprise finding:** Dù Naive Baseline có Faithfulness ban đầu tương đối (0.7917), nhưng Answer Relevancy lại rất thấp (0.3967) do hệ thống search trả về các raw chunk chưa thật sự khớp với Intent của câu hỏi. Khi thêm Rerank, Recall và Relevancy đều tăng vọt.

## Presentation Notes (5 phút)

1. RAGAS scores (naive vs production): Production RAG đã cải thiện toàn diện cả 4 chỉ số, chứng minh giá trị của Pipeline đầy đủ.
2. Biggest win — module nào, tại sao: M2 (Hybrid Search) & M3 (Reranking) vì các module này loại bỏ chunk rác (tăng Precision) và không bỏ sót thông tin quan trọng (tăng Recall).
3. Case study — 1 failure, Error Tree walkthrough: Câu hỏi "Kỳ tính thuế trong tờ khai thuế GTGT...". Điểm Context Precision thấp (0.33) do có nhiều chunk không liên quan. Root cause: Thiếu Metadata Filtering. Fix: Bổ sung PreRAG filter theo document type.
4. Next optimization nếu có thêm 1 giờ: Tích hợp Metadata Filtering để lọc chính xác văn bản (chỉ tìm trong Tờ khai thuế) và thử nghiệm các mô hình Embeddings hỗ trợ tiếng Việt tốt hơn.
