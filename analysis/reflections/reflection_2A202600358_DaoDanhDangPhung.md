# Individual Reflection — Lab 18

**Tên:** Đào Danh Đăng Phụng  
**Module phụ trách:** M5 - Enrichment Pipeline

---

## 1. Đóng góp kỹ thuật

- Module đã implement: `src/m5_enrichment.py`
- Các hàm/class chính đã viết:
  - `summarize_chunk()` để tóm tắt chunk bằng OpenAI hoặc fallback extractive.
  - `generate_hypothesis_questions()` để sinh câu hỏi HyQA giúp giảm vocabulary gap.
  - `contextual_prepend()` để thêm ngữ cảnh tài liệu trước chunk gốc.
  - `extract_metadata()` để tự động tạo metadata gồm topic, entities, category, language.
  - `enrich_chunks()` để chạy pipeline enrichment và trả về `EnrichedChunk`.
- Số tests pass: 10/10 cho `pytest tests/test_m5.py -v`

## 2. Kiến thức học được

- Khái niệm mới nhất: enrichment trước khi embedding giúp retrieval có thêm tín hiệu ngoài raw text.
- Điều bất ngờ nhất: HyQA có thể bridge khoảng cách giữa cách người dùng hỏi và cách tài liệu viết.
- Kết nối với bài giảng: M5 nằm ở bước Pre-RAG/Augmentation, cải thiện chất lượng chunk trước khi index.

## 3. Khó khăn & Cách giải quyết

- Khó khăn lớn nhất: cần module chạy được cả khi không có OpenAI API key.
- Cách giải quyết: implement OpenAI path dạng optional và fallback deterministic không cần network.
- Thời gian debug: kiểm tra output type, preservation của original text, và test M5.

## 4. Nếu làm lại

- Sẽ làm khác điều gì: thêm cache cho enrichment để tránh gọi LLM lại nhiều lần khi indexing.
- Module nào muốn thử tiếp: M2 Hybrid Search, vì M5 hiệu quả nhất khi phần index/search dùng tốt enriched text và metadata.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 4 |
| Code quality | 4 |
| Teamwork | 4 |
| Problem solving | 4 |
