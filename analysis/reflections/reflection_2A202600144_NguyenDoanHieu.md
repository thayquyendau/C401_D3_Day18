# Individual Reflection - Lab 18

**Tên:** Người 2  
**Module phụ trách:** M2 - Hybrid Search

---

## 1. Đóng góp kỹ thuật

- Module đã implement: `src/m2_search.py`
- Các hàm/class chính đã viết:
  - `segment_vietnamese()` để tách từ tiếng Việt bằng `underthesea.word_tokenize(text, format="text")`, giúp BM25 nhận diện các cụm từ tốt hơn.
  - `BM25Search.index()` để segment từng document, tokenize bằng khoảng trắng và tạo chỉ mục `BM25Okapi`.
  - `BM25Search.search()` để segment query, tính điểm BM25 và trả về top-k `SearchResult` với `method="bm25"`.
  - `DenseSearch.index()` để encode chunks bằng model embedding `BAAI/bge-m3` và upsert vector vào Qdrant.
  - `DenseSearch.search()` để encode query và tìm kiếm dense vector trong Qdrant.
  - `reciprocal_rank_fusion()` để gộp kết quả BM25 và Dense Search bằng công thức RRF.
  - `HybridSearch` để đóng gói pipeline search gồm BM25, Dense Search và RRF.
- Số tests pass: 5/5 cho `pytest tests/test_m2.py -v`

## 2. Kiến thức học được

- Khái niệm mới nhất: Hybrid Search kết hợp keyword retrieval và semantic retrieval để tận dụng điểm mạnh của cả BM25 lẫn embedding.
- Điều bất ngờ nhất: với tiếng Việt, word segmentation ảnh hưởng rõ đến BM25 vì các cụm như "nhân viên" hoặc "nghỉ phép" có thể cần được xử lý như một đơn vị ý nghĩa.
- Kết nối với bài giảng: M2 nằm ở bước retrieval trong RAG pipeline; chất lượng retrieval quyết định context nào được đưa sang reranker và generation.

## 3. Khó khăn & Cách giải quyết

- Khó khăn lớn nhất: cần làm cho BM25, Dense Search và Hybrid Search có cùng format output để pipeline phía sau sử dụng được.
- Cách giải quyết: dùng chung dataclass `SearchResult` gồm `text`, `score`, `metadata`, `method`; mỗi search method chỉ cần trả về cùng một kiểu dữ liệu.
- Khó khăn về dependency: Dense Search phụ thuộc vào Qdrant và model embedding, nên cần `docker-compose.yml`, `qdrant-client`, `sentence-transformers` và Qdrant chạy trên port `6333`.
- Thời gian debug: chủ yếu kiểm tra return type, thứ tự ranking BM25 và logic RRF để đảm bảo kết quả hybrid có `method="hybrid"`.

## 4. Nếu làm lại

- Sẽ làm khác điều gì: thêm smoke test riêng cho `DenseSearch` với Qdrant local để đảm bảo pipeline nhóm không chỉ pass BM25/RRF mà còn chạy được dense retrieval thật.
- Cải tiến tiếp theo: lưu `chunk_id` rõ ràng trong metadata để RRF gộp kết quả theo id thay vì chỉ gộp theo text, tránh trùng lặp hoặc mất metadata khi hai chunks có nội dung giống nhau.
- Module muốn thử tiếp: M3 Reranking, vì reranker là bước trực tiếp cải thiện chất lượng top-k sau khi M2 đã retrieve nhiều ứng viên.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 4 |
| Code quality | 4 |
| Teamwork | 5 |
| Problem solving | 4 |
