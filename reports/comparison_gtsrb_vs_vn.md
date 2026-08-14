# So sánh: Baseline GTSRB vs Finetuned VN

## 1. Baseline GTSRB (chưa finetune) trên VN test
- Mean max confidence: **0.5228**
- Số ảnh test VN: 1712
- *Lưu ý*: Không tính được accuracy trực tiếp do label space khác (43 vs 56). Confidence thấp = model 'lúng túng' khi gặp biển VN.

## 2. Finetuned trên VN test
- Accuracy: **0.8925**
- Top-3 accuracy: **0.9778**
- Loss: 0.3308
- Số lớp VN: 56

## 3. Kết luận
Fine-tune 2-stage cải thiện rõ rệt khả năng nhận diện biển VN so với model GTSRB gốc, nhờ tận dụng feature extractor block1+2 (kiến thức chung về biển báo theo Vienna 1968) và học lại block3 + head cho QCVN 41:2019.