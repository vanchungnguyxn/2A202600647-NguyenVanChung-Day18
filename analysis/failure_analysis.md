# Failure Analysis — Lab 18: Production RAG

**Nhóm:** Cá nhân  
**Thành viên:** NguyenVanChung (M1–M5)

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.8833 | 0.9091 | +0.0258 |
| Answer Relevancy | 0.8033 | 0.8582 | +0.0549 |
| Context Precision | 0.9250 | 1.0000 | +0.0750 |
| Context Recall | 0.9083 | 0.9500 | +0.0417 |

**Kết luận:** Production pipeline vượt baseline trên cả 4 metrics. Faithfulness ≥ 0.85, tất cả metrics ≥ 0.75. Context Precision đạt 1.0.

## Bottom-5 Failures

### #1
- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Phí 2%/tháng trên 15M ≈ 50.000 VNĐ cho 5 ngày quá hạn (pro-rata).
- **Got:** 50.000 VNĐ (đúng con số) nhưng LLM tự suy luận từng bước tính toán → faithfulness thấp (0.18).
- **Worst metric:** faithfulness (0.18)
- **Error Tree:** Output sai → Context đúng? (recall=1.0) → LLM tự tính toán sai số học
- **Root cause:** Model tự suy luận phép tính pro-rata thay vì trích đúng từ context.
- **Suggested fix:** Prompt yêu cầu “không tự tính nếu context không có công thức; nêu công thức từ tài liệu”.

### #2
- **Question:** Có cần kích hoạt MFA không?
- **Expected:** Có, bắt buộc theo v2.0; v1.0 không yêu cầu.
- **Got:** Có, bắt buộc (đúng phần chính, thiếu so sánh v1.0).
- **Worst metric:** context_recall (0.50)
- **Error Tree:** Output đúng → Context thiếu chunk policy v1.0 → recall thấp
- **Root cause:** Retrieve không lấy cả 2 phiên bản chính sách mật khẩu.
- **Suggested fix:** Metadata filter theo version + HyQA “chính sách cũ vs mới”.

### #3
- **Question:** Tài trợ khóa học 25 triệu, nghỉ sau 8 tháng — hoàn trả bao nhiêu?
- **Expected:** 100% = 25 triệu + nêu cam kết 1 năm.
- **Got:** 25 triệu (đúng số tiền, thiếu lý do cam kết).
- **Worst metric:** faithfulness (0.50)
- **Error Tree:** Output đúng số → Thiếu điều kiện cam kết trong câu trả lời
- **Root cause:** Prompt chưa bắt buộc nêu điều kiện/ngoại lệ kèm con số.
- **Suggested fix:** Prompt template: “kèm điều kiện áp dụng nếu có trong context”.

### #4
- **Question:** Bao lâu phải đổi mật khẩu một lần?
- **Expected:** 120 ngày (v2.0), chính sách cũ 90 ngày đã thay thế.
- **Got:** 120 ngày theo v2.0 (đúng, thiếu mention v1.0).
- **Worst metric:** context_precision (0.50)
- **Error Tree:** Output đúng → Context có chunk nhiễu (v1.0 + v2.0)
- **Root cause:** Corpus có nhiều version; precision bị phạt vì chunk cũ vẫn được retrieve.
- **Suggested fix:** Enrichment metadata `version` + filter ưu tiên `superseded=false`.

### #5
- **Question:** Nghỉ phép không lương 20 ngày cần ai phê duyệt?
- **Expected:** CEO + lưu ý tự đóng bảo hiểm nếu >14 ngày.
- **Got:** CEO phê duyệt (thiếu lưu ý bảo hiểm).
- **Worst metric:** faithfulness (0.50)
- **Error Tree:** Output đúng phần phê duyệt → Thiếu khía cạnh bảo hiểm trong context/answer
- **Root cause:** Multi-aspect question; retrieve/answer chỉ cover 1 khía cạnh.
- **Suggested fix:** HyQA tạo câu hỏi phụ “ảnh hưởng bảo hiểm khi nghỉ không lương dài”.

## Case Study (cho presentation)

**Question chọn phân tích:** “Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?”

**Error Tree walkthrough:**
1. Output đúng? → Con số đúng (~50.000 VNĐ) nhưng reasoning tự bịa bước tính
2. Context đúng? → Có (recall=1.0, precision=1.0)
3. Query rewrite OK? → OK
4. Fix ở bước: **LLM generation** — cấm tự tính toán, bắt trích công thức từ context

**Nếu có thêm 1 giờ, sẽ optimize:**
- Thêm rule “numeric answers must quote formula from context” trong system prompt.
