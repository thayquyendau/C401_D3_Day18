# Individual Reflection — Lab 18

**Tên:** Người 1  
**Module phụ trách:** M1 - Advanced Chunking Strategies

---

## 1. Đóng góp kỹ thuật

- Module đã implement: **M1 - Advanced Chunking**
- Các hàm/class chính đã viết:
  - `chunk_semantic()`: Semantic chunking dựa trên cosine similarity giữa các câu
  - `chunk_hierarchical()`: Parent-child chunking với parent 2048 chars, child 256 chars
  - `chunk_structure_aware()`: Structure-aware chunking theo markdown headers
  - `compare_strategies()`: So sánh 4 strategies (basic + 3 advanced)
- Số tests pass: **13/13** ✅
- **Data preparation:**
  - Sử dụng Mistral OCR API để extract PDF → Markdown
  - Tạo dữ liệu test thực: BCTC.md (tờ khai thuế), Nghi_dinh_so_13-2023_ve_bao_ve_du_lieu_ca_nhan_508ee (văn bản pháp luật)
  - Viết script `test_chunking_real_data.py` để test với dữ liệu tiếng Việt thực tế

## 2. Kiến thức học được

- **Khái niệm mới nhất:**
  - Semantic chunking: Nhóm câu theo similarity thay vì cắt cứng theo size
  - Hierarchical chunking: Pattern retrieve child (precision) → return parent (context)
  - Structure-aware: Giữ nguyên logical structure của document
  
- **Điều bất ngờ nhất:**
  - Semantic chunking với threshold 0.85 tạo ra ít chunks hơn basic chunking
  - Hierarchical pattern rất hiệu quả: index children nhỏ (embedding chính xác), nhưng trả parent lớn (đủ context cho LLM)
  - sentence-transformers model "all-MiniLM-L6-v2" rất nhanh và đủ tốt cho Vietnamese text
  - **Testing với real data:** Structure-aware chunking xuất sắc cho văn bản pháp luật (78 sections từ Nghị định), hierarchical chunking cân bằng nhất cho production
  - Semantic chunking với threshold thấp (0.7) tạo quá nhiều chunks nhỏ (avg 66 chars) - cần tune threshold
  
- **Kết nối với bài giảng:**
  - Chunking strategies ảnh hưởng trực tiếp đến Context Precision và Context Recall trong RAGAS
  - Hierarchical chunking là best practice cho production RAG systems

## 3. Khó khăn & Cách giải quyết

- **Khó khăn lớn nhất:**
  - Semantic chunking cần xử lý edge cases (empty text, single sentence)
  - Structure-aware chunking cần handle documents không có headers
  
- **Cách giải quyết:**
  - Thêm validation checks cho empty inputs
  - Regex pattern `^#{1,3}\s+` để match markdown headers level 1-3
  
- **Thời gian debug:** ~15 phút (chủ yếu fix type hints)

## 4. Nếu làm lại

- **Sẽ làm khác điều gì:**
  - Test với Vietnamese text nhiều hơn để tune semantic threshold
  - Thử các embedding models khác (bge-m3, multilingual-e5)
  - Add caching cho sentence embeddings để tăng tốc
  
- **Module nào muốn thử tiếp:**
  - M2 (Hybrid Search): Muốn thử implement BM25 với Vietnamese segmentation
  - M3 (Reranking): Cross-encoder reranking có vẻ thú vị

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 5 |
| Code quality | 5 |
| Teamwork | 5 |
| Problem solving | 5 |
