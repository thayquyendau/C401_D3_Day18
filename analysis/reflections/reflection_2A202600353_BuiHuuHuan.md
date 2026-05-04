# Reflection - Bui Huu Huan

## 1. Dong gop ky thuat cu the

- Phu trach chinh M4 Evaluation.
- Implement `evaluate_ragas` trong `src/m4_eval.py` voi du 4 metrics: faithfulness, answer_relevancy, context_precision, context_recall.
- Implement real RAGAS evaluation path and make missing API/model dependencies fail clearly.
- Implement `failure_analysis` voi diagnostic mapping tu metric thap sang root cause va suggested fix.
- Implement `save_report` tao `reports/ragas_report.json` hop le va serialize sach.
- Ho tro tich hop `src/pipeline.py` de chay end-to-end va sinh `analysis/failure_analysis.md`.

## 2. Kien thuc hoc duoc

- Cach danh gia Production RAG bang RAGAS va y nghia cua tung metric.
- Faithfulness kiem tra cau tra loi co bam context hay khong.
- Answer relevancy do muc do cau tra loi khop intent cau hoi.
- Context precision do do nhieu cua retrieved contexts.
- Context recall do kha nang retrieve du context chua dap an.
- Failure analysis giup bien diem so thanh hanh dong sua loi o chunking, search, rerank, generation.

## 3. Kho khan va cach giai quyet

- RAGAS va LLM phu thuoc API key nen can cau hinh GitHub Models/OpenAI ro rang truoc khi chay.
- Context tieng Viet co dau va du lieu PDF kho parse nen dung token normalization va overlap scoring.
- JSON report can hop le nen convert tat ca scores ve Python float va tranh object khong serialize duoc.
- `test_set.json` co trailing comma nen loader duoc lam khoan dung hon.

## 4. Tu danh gia

- Tu danh gia: 5/5.
- Ly do: hoan thanh M4, tich hop pipeline end-to-end, co JSON report, failure analysis, va tests M4 pass.
