# Failure Analysis — Lab 18: Production RAG

**Nhóm:** Nhóm D3  
**Thành viên:** [Tên 1 → M1] · [2A202600144_Nguyễn Doãn Hiếu → M2] · [2A202600359_Đậu Văn Quyền → M3] · [Tên 4 → M4]

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.7917 | 0.9444 | +0.1527 |
| Answer Relevancy | 0.3967 | 0.8475 | +0.4508 |
| Context Precision | 0.7083 | 0.8240 | +0.1157 |
| Context Recall | 0.6944 | 0.9108 | +0.2164 |

## Bottom-5 Failures

### #1
- **Question:** Cơ quan chuyên trách bảo vệ dữ liệu cá nhân là cơ quan nào?
- **Expected:** Cục An ninh mạng và phòng, chống tội phạm sử dụng công nghệ cao.
- **Got:** (Câu trả lời bị sai lệch hoặc không có căn cứ từ context)
- **Worst metric:** faithfulness (0.5)
- **Error Tree:** Output sai → Context đúng? Có → Query OK? Có → Lỗi ở Generator.
- **Root cause:** LLM hallucinating hoặc prompt chưa đủ chặt chẽ khiến LLM sinh ra thông tin không có trong tài liệu.
- **Suggested fix:** Tighten prompt (bắt buộc chỉ dùng context), lower temperature.

### #2
- **Question:** Chủ thể dữ liệu có nghĩa vụ gì theo Nghị định 13?
- **Expected:** Liệt kê đầy đủ các nghĩa vụ theo Nghị định.
- **Got:** Trả lời thiếu một số nghĩa vụ.
- **Worst metric:** context_recall (0.4)
- **Error Tree:** Output sai → Context đúng? Không (bị thiếu chunk) → Lỗi ở Retriever.
- **Root cause:** Missing relevant chunks do Hybrid Search chưa lấy đủ context chứa toàn bộ nghĩa vụ (nghĩa vụ nằm rải rác).
- **Suggested fix:** Improve chunking (tăng chunk size) hoặc tăng top_k.

### #3
- **Question:** Kỳ tính thuế trong tờ khai thuế GTGT của DHA Surfaces là kỳ nào?
- **Expected:** Câu trả lời tương ứng với thông tin trong tờ khai.
- **Got:** Trả về kèm nhiều thông tin nhiễu từ các văn bản khác.
- **Worst metric:** context_precision (0.33)
- **Error Tree:** Output đúng/sai → Context đúng? Không (có quá nhiều chunk rác) → Lỗi ở Reranker / Retriever.
- **Root cause:** Too many irrelevant chunks, hệ thống bị nhầm lẫn giữa nhiều tài liệu.
- **Suggested fix:** Add metadata filter (chỉ tìm trong file tờ khai).

### #4
- **Question:** Nghị định 13/2023/NĐ-CP áp dụng cho những đối tượng nào?
- **Expected:** Cơ quan, tổ chức, cá nhân có liên quan.
- **Got:** Bịa thêm đối tượng không thuộc luật định.
- **Worst metric:** faithfulness (0.67)
- **Error Tree:** Output sai → Context đúng? Có → Lỗi ở Generator.
- **Root cause:** LLM hallucinating.
- **Suggested fix:** Tighten prompt, lower temperature = 0.

### #5
- **Question:** Khi nào có thể xử lý dữ liệu cá nhân mà không cần sự đồng ý của chủ thể dữ liệu?
- **Expected:** Khẩn cấp, bảo vệ tính mạng, v.v.
- **Got:** Liệt kê thiếu.
- **Worst metric:** context_recall (0.4)
- **Error Tree:** Output sai → Context đúng? Không (thiếu chunk) → Lỗi ở Retriever.
- **Root cause:** Missing relevant chunks (thông tin trải dài trên nhiều mục).
- **Suggested fix:** Improve chunking, dùng BM25 weight cao hơn.

## Case Study (cho presentation)

**Question chọn phân tích:** Kỳ tính thuế trong tờ khai thuế GTGT của DHA Surfaces là kỳ nào?

**Error Tree walkthrough:**
1. Output đúng? → Có thể đúng nhưng context kèm theo rác.
2. Context đúng? → Không, có quá nhiều irrelevant chunks làm loãng context.
3. Query rewrite OK? → Query rõ ràng nhưng thiếu cơ chế lọc tài liệu.
4. Fix ở bước: Thêm Pre-RAG để lọc theo metadata `source` hoặc `document_type`.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Tích hợp Self-Query Retrieval để tự động nhận diện ý định "Tờ khai thuế" và add metadata filter vào lệnh search Qdrant.
