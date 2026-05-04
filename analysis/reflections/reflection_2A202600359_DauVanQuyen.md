## Individual Reflection - Lab 18

**Tên:** Đậu Văn Quyền
**Module phụ trách:** M3

---

## 1. Đóng góp kỹ thuật

- Module đã implement: M3 Reranking.
- Các hàm/class chính đã viết: `CrossEncoderReranker._load_model()`, `CrossEncoderReranker.rerank()`, `benchmark_reranker()`.
- Số tests pass: 5/5 khi chạy bằng `.venv` với `USE_REAL_RERANKER=1`.

## 2. Kiến thức học được

- Khái niệm mới nhất: cross-encoder reranking khác với retrieval score ordering ở chỗ model chấm điểm trực tiếp trên cặp `(query, document)` thay vì chỉ dựa vào score truy hồi ban đầu.
- Điều bất ngờ nhất: việc xác minh model thật phụ thuộc rất nhiều vào môi trường chạy, cache model và kết nối đến HuggingFace, không chỉ phụ thuộc vào logic code.
- Kết nối với bài giảng: phần reranking trong pipeline Production RAG, top-k retrieval -> rerank -> context cuối cùng cho generation.

## 3. Khó khăn & Cách giải quyết

- Khó khăn lớn nhất: model thật trong `.venv` chạy chậm ở lần đầu và bị retry do không kết nối được đến HuggingFace (`WinError 10061`), trong khi chạy bằng Python global trước đó lại không đúng môi trường có `FlagEmbedding`.
- Cách giải quyết: chuẩn hóa việc chạy test bằng `D:\VIN_University\Giaidoan2Track3\Day18-Track3-Production-RAG\.venv\Scripts\python.exe`, tách riêng test `test_rerank_returns` để quan sát quá trình load model, sau đó chạy full suite để xác nhận kết quả cuối cùng. Code giữ fallback an toàn nếu model thật không load được.
- Thời gian debug: khoảng 20-30 phút, chủ yếu để phân biệt vấn đề môi trường Python và độ trễ khi model khởi tạo/tải về.

## 4. Nếu làm lại

- Sẽ làm khác điều gì: thêm logging rõ hơn cho backend đang được sử dụng (`FlagReranker`, `CrossEncoder`, hay fallback lexical) và hướng dẫn cache model local để giảm thời gian test lần đầu.
- Module nào muốn thử tiếp: M2 hoặc M4, vì chất lượng reranking liên quan trực tiếp đến retrieval và evaluation.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 4 |
| Code quality | 4 |
| Teamwork | 4 |
| Problem solving | 5 |
