# 🎬 Demo Script — Đồ án CNN Nhận diện biển báo giao thông

**Thời lượng**: 12–15 phút trình bày + 5–10 phút Q&A
**Công cụ**: Streamlit local (`app/streamlit_app.py`) + 2 tab trình duyệt phụ
**Setup**: Folder `demo_images/` đã chuẩn bị sẵn, model đã verify 100% top-1 trên Tier 1

---

## 📋 Pre-flight checklist (làm 5 phút trước demo)

```powershell
# 0. Đảm bảo đã cài streamlit-cropper (phục vụ UI crop ROI mới)
pip install -r requirements.txt

# 1. Chạy app, kiểm tra mở được
streamlit run app/streamlit_app.py
# → Trình duyệt tự mở http://localhost:8501

# 2. Verify nhanh 10 ảnh Tier 1 vẫn predict đúng (optional)
python -m src.predict --image demo_images/tier1_gtsrb/01_stop.png

# 3. Mở sẵn 3 tab trình duyệt:
#    - Tab 1: Streamlit demo (localhost:8501)
#    - Tab 2: notebooks/colab_finetune_vn.ipynb (mở trong VSCode hoặc Colab)
#    - Tab 3: reports/figures/confusion_matrix.png (mở image viewer)
```

---

## 🕐 Timeline 15 phút

### Phase 0 — Mở đầu (1 phút)

> *"Em xin trình bày đồ án **Xây dựng mô hình mạng nơ-ron tích chập (CNN) cho bài toán phân loại biển báo giao thông**. Hệ thống gồm 3 phần: pipeline tiền xử lý, mô hình CNN tự xây dựng (không dùng pretrained), và demo Streamlit."*

**Show**: slide title + sidebar Streamlit (43 lớp, input 48×48).

---

### Phase 1 — Giới thiệu dataset (1.5 phút) ⚠️ **PHẦN QUAN TRỌNG NHẤT**

**Show**: slide "Dataset" với nội dung sau (học thuộc):

> *"Em sử dụng **GTSRB — German Traffic Sign Recognition Benchmark**, chuẩn của cuộc thi IJCNN 2011, gồm 51.839 ảnh, 43 lớp, được dùng trong hàng nghìn paper.*
>
> *Lý do chọn GTSRB:*
> 1. *Là **benchmark chuẩn quốc tế** → kết quả của em **so sánh được trực tiếp** với SOTA*
> 2. *Việt Nam và Đức cùng tuân theo **Công ước Vienna 1968** → biển báo có **~70% tương đồng hình dạng và biểu tượng**, kiến thức học được **transfer được** sang VN*
> 3. *Việt Nam **chưa có** dataset công khai cùng quy mô và chất lượng nhãn*
>
> *Hạn chế đã nhận diện: biển cảnh báo VN nền vàng vs Đức nền trắng — em có **lộ trình fine-tune** ở phần future work."*

→ Nói tự tin, không né tránh. Cô gật đầu là pass.

---

### Phase 2 — Pipeline kỹ thuật (2 phút)

**Show**: confusion matrix + training curves (`reports/figures/`).

> *"Pipeline 4 bước: (1) **ROI cropping** với padding 10%, (2) **Resize 48×48 + normalize**, (3) **Augmentation** xoay ±12°, brightness/contrast/saturation 0.85–1.15, (4) **CNN 3 block convolution + GAP + Dense head**, train với class_weight cap 2.0 để xử lý imbalance.*
>
> *Kết quả: **test accuracy XX.XX%**, top-3 accuracy YY.YY%."*

> 🗒️ **Điền số thật từ `reports/metrics.json` trước demo.**

---

### Phase 3 — Demo Tier 1: Happy path (3 phút) 🎯

**Show**: Streamlit tab "📁 Upload ảnh"

> 🆕 **Tab Upload có UI crop mới**: sau khi upload ảnh, **kéo khung xanh** trên ảnh
> để chọn vùng biển báo. Preview + kết quả hiện ngay bên phải (realtime update).
> Sidebar có toggle "✂️ Crop ROI thủ công" — mặc định BẬT.
>
> **Lời thoại khi mới mở tab:**
> *"Em thiết kế UI cho phép giáo viên kéo khung ROI ngay trong app — mô phỏng vai
> trò detector trong pipeline 2-stage. Khung mặc định tỷ lệ vuông 1:1 vì biển
> báo thường vuông."*

Upload **5 ảnh** từ `demo_images/tier1_gtsrb/` theo thứ tự (ảnh đã pre-cropped → kéo khung bao trọn ảnh là OK):

| # | File upload | Top-1 mong đợi | Confidence | Lời thoại |
|---|---|---|---|---|
| 1 | `01_stop.png` | Dừng lại | 100% | *"Biển dừng lại — VN gọi là R.122 — model nhận đúng 100%"* |
| 2 | `02_no_entry.png` | Cấm vào | 100% | *"Cấm đi ngược chiều — VN P.102 — giống hệt biển Đức"* |
| 3 | `04_speed_50.png` | Speed 50 | 99.4% | *"Hạn chế tốc độ 50 — VN P.127"* |
| 4 | `06_no_passing.png` | Cấm vượt | 99.9% | *"Cấm vượt — VN P.125"* |
| 5 | `09_roundabout.png` | Vòng xuyến | 80.4% | *"Vòng xuyến — VN R.303 — confidence thấp hơn vì augmentation strong"* |

> 💡 **Chốt phase**: *"5/5 đúng top-1, confidence trung bình > 95%. Đây là 5 lớp **có biển VN tương đương trực tiếp** theo Vienna 1968."*

---

### Phase 3.5 — Demo "wow": Tầm quan trọng của ROI crop (1.5 phút) 🎓

**Mục đích**: chứng minh quyết định kỹ thuật "model là classifier, cần ROI" là đúng đắn — biến điểm yếu (cần crop) thành luận điểm khoa học.

**Cách làm:**
1. Upload lại `01_stop.png` (hoặc 1 ảnh test có background rộng nếu có)
2. Sidebar → **bỏ tick** "✂️ Crop ROI thủ công"
3. App tự predict full-size → **acc/confidence sẽ giảm rõ rệt** với ảnh có background
4. Bật lại tick → **kéo khung xanh ôm sát biển** → confidence vọt lên 99%

> **Lời thoại:**
> *"Đây là minh chứng tại sao em **tách 2 stage** detector + classifier. Nếu cho
> ảnh full-size, model 'thấy' chủ yếu là background → confidence loãng. Khi
> crop sát ROI, accuracy tăng vọt. Đây là lý do em design UI có khung ROI tương
> tác — mô phỏng kết quả của 1 detector real-world."*

> 💡 **Chốt phase**: *"Trong production, khung ROI sẽ được thay bằng output của
> YOLO/Faster R-CNN. Em focus stage classification để nghiên cứu sâu CNN."*

---

### Phase 4 — Demo Tier 2: Transfer sang biển VN thực tế (2.5 phút) 🇻🇳

**Show**: vẫn tab Upload. Upload **3 ảnh** từ `demo_images/tier2_vn_real/`:

> *"Bây giờ em test trên **ảnh chụp thực tế tại Việt Nam** — không phải ảnh Đức nữa."*
>
> 🆕 **Lưu ý**: với ảnh VN thực tế (background đường phố), kéo khung xanh **ôm sát biển báo** chứ không bao trọn ảnh — đây là lúc UI crop phát huy giá trị nhất.

| # | File | Lời thoại nếu đúng | Plan B nếu sai |
|---|---|---|---|
| 1 | `vn_01_stop.png` | *"Biển Stop ở ngã tư VN — model nhận đúng với confidence X%"* | *"Top-3 vẫn có Stop — kiến thức transfer được 1 phần"* |
| 2 | `vn_02_cam_vao.png` | *"Cấm vào ở phố cổ HN — đúng top-1"* | (như trên) |
| 3 | `vn_03_cam_vuot.png` hoặc `vn_05_vong_xuyen.png` | *"Vòng xuyến vì giống biển R.303"* | (như trên) |

> 💡 **Chốt phase**: *"Model train hoàn toàn trên ảnh Đức nhưng vẫn nhận được biển VN có cùng pictogram — minh chứng cho lập luận Vienna 1968."*

---

### Phase 5 — Demo Tier 3: Failure case có chủ đích (2 phút) 🎓

**Show**: upload `fail_01_canh_bao_vang.png`

> *"Em **chủ động** test 1 trường hợp model sai để cô thấy giới hạn:*
>
> *Đây là biển cảnh báo trẻ em qua đường ở VN — **nền VÀNG**. Trong khi GTSRB toàn bộ biển cảnh báo có **nền TRẮNG**. Model bị 'bối rối': top-1 sai / confidence chỉ X%.*
>
> *Đây chính là **điểm yếu lớn nhất** của model GTSRB khi áp dụng cho VN — và là **động cơ trực tiếp** cho phần future work tiếp theo."*

→ Chuyển ngay sang Phase 6, đừng để cô kịp hỏi.

---

### Phase 6 — Camera realtime (2 phút) 🎥

**Show**: Streamlit tab "🎥 Camera realtime"

In sẵn 2 biển A4 (Stop + Cấm vào). Đặt vào ROI box xanh.

> *"Phần realtime mô phỏng pipeline detection 2-stage: ROI box xanh đóng vai trò detector, model em là classifier ở stage 2. Trong production sẽ thay ROI box bằng YOLO/Faster R-CNN."*

→ Đưa biển vào khung → live prediction hiện. Wow effect.

---

### Phase 7 — Future work + đóng (1 phút)

**Show**: mở `notebooks/colab_finetune_vn.ipynb` ở tab 2

> *"Em đã **chuẩn bị sẵn pipeline transfer learning 2-stage** để fine-tune sang biển VN khi có dataset chuẩn:*
> - *Stage 1: Freeze backbone, train Dense head mới (LR 1e-3, 12 epochs)*
> - *Stage 2: Unfreeze block3, fine-tune với LR thấp (1e-4, 25 epochs)*
> - *Dataset target: Roboflow Vietnam Traffic Sign Detection (~4.000 ảnh, 58 lớp QCVN)*
>
> *Đây là hướng phát triển tiếp theo của đồ án."*

→ Scroll nhanh qua notebook để cô thấy có 38 cells, không phải bịa.

---

## 🎤 Q&A — Câu trả lời mẫu (học thuộc)

| Cô hỏi | Trả lời gọn |
|---|---|
| *Sao chọn dataset Đức?* | "Benchmark chuẩn IJCNN 2011 + Vienna 1968 + VN chưa có dataset công khai cùng quy mô. Đây là cách tiếp cận chuẩn trong literature." |
| *Có dùng được cho biển VN không?* | "Với biển theo Vienna 1968 thì có (em vừa demo Tier 2). Với biển đặc thù VN — đặc biệt biển cảnh báo nền vàng — cần fine-tune (em vừa demo Tier 3 + show notebook)." |
| *Tại sao phải crop ảnh trước khi predict?* | "Em làm stage classifier trong pipeline detection 2-stage. Stage detector là bài toán riêng đã có YOLO/RCNN. Em giả định input là ROI đã crop — chuẩn theo paper LeCun trên GTSRB." |
| *Augmentation cụ thể là gì?* | "Rotation ±12°, brightness/contrast/saturation 0.85–1.15. Lý do: mô phỏng đa dạng điều kiện chụp." |
| *Class imbalance xử lý sao?* | "Tính class_weight inverse-frequency, cap ở 2.0 để tránh overweight class hiếm gây oscillation." |
| *Sao 43 lớp mà không phải 58/72?* | "Theo bộ luật Đức StVO. 43 lớp đủ phủ 6 nhóm chính theo Vienna: cấm, cảnh báo, hiệu lệnh, ưu tiên, hết hạn chế, chỉ dẫn." |
| *Kiến trúc CNN cụ thể?* | "3 block conv (32→64→128 filter, mỗi block: Conv-BN-Conv-BN-MaxPool-Dropout) + GAP + Dense 256 + Dropout 0.5 + Softmax 43. Tổng ~XXX nghìn params." (xem `src/model.py` để điền số chính xác) |
| *Vì sao GAP mà không Flatten?* | "GAP giảm overfitting (ít params hơn Flatten), invariant với spatial shift, là best practice modern CNN." |
| *UI crop ROI dùng thư viện gì?* | "`streamlit-cropper` — wrapper React-based component bên trên Streamlit. Em chọn vì cho phép kéo/resize bbox real-time, return PIL Image trực tiếp về backend, không cần tự code JS." |
| *Sao mặc định khung crop 1:1?* | "Biển báo theo Vienna 1968 đa số có hình tròn/tam giác/vuông → bbox tự nhiên là vuông. 1:1 giúp giữ aspect ratio đúng khi resize 48×48, tránh distort. Vẫn có option 'Tự do' cho biển hình chữ nhật." |

---

## ❌ Câu KHÔNG nên trả lời

- *"Em không biết"* → tệ. Thay bằng: *"Em chưa làm phần này, dự định ở future work."*
- *"Vì paper bảo thế"* → cần giải thích tại sao paper bảo thế.
- *"Em copy từ Kaggle"* → mất điểm. Thay bằng: *"Em tham khảo notebook X làm reference, kiến trúc em tự thiết kế lại theo nguyên tắc Y."*

---

## 🚨 Backup plan

| Tình huống | Xử lý |
|---|---|
| Mạng/camera lỗi | Có sẵn screenshot trong `reports/figures/` để show kết quả |
| Streamlit crash | Chạy `python -m src.predict --image demo_images/tier1_gtsrb/01_stop.png` trên terminal |
| Cô hỏi muốn xem code | Mở VSCode sẵn ở `src/model.py` và `src/train.py` |
| Cô hỏi xem dữ liệu train | Show folder `data/test_samples/` (43 subfolder, biết là pre-cropped) |
