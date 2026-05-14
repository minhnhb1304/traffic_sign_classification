# Demo Images — Hướng dẫn chuẩn bị

Folder chứa ảnh demo cho buổi bảo vệ. Cấu trúc 3 tier theo độ tin cậy + mục đích.

```
demo_images/
├── tier1_gtsrb/      # 10 ảnh GTSRB pre-cropped — demo "happy path" (đã có sẵn)
├── tier2_vn_real/    # Ảnh biển VN thực tế — demo "transfer ok" (BẠN TỰ LẤY)
└── tier3_failure/    # Failure case có chủ đích — demo liêm chính (BẠN TỰ LẤY)
```

---

## Tier 1 — `tier1_gtsrb/` ✅ ĐÃ CHUẨN BỊ SẴN

10 ảnh chọn lọc từ `data/test_samples/`, **đã verify 10/10 top-1 đúng**, confidence trung bình ~97%.

| File | Lớp dự đoán | Confidence verify |
|---|---|---|
| `01_stop.png` | Dừng lại | 100.00% |
| `02_no_entry.png` | Cấm vào | 100.00% |
| `03_speed_30.png` | Giới hạn tốc độ (30km/h) | 94.68% |
| `04_speed_50.png` | Giới hạn tốc độ (50km/h) | 99.44% |
| `05_speed_60.png` | Giới hạn tốc độ (60km/h) | 99.74% |
| `06_no_passing.png` | Cấm vượt | 99.99% |
| `07_priority_road.png` | Đường ưu tiên | 100.00% |
| `08_yield.png` | Nhường đường | 99.75% |
| `09_roundabout.png` | Bắt buộc đi vòng xuyến | 80.35% |
| `10_ahead_only.png` | Chỉ được đi thẳng | 100.00% |

**Tiêu chí chọn**: cả 10 lớp đều có **biển báo Việt Nam tương đương** theo QCVN 41:2019 (Vienna Convention 1968) → an toàn để demo, dễ trả lời câu hỏi "ứng dụng cho VN".

---

## Tier 2 — `tier2_vn_real/` 🔧 BẠN CẦN TỰ LẤY (5 ảnh)

**Mục đích**: chứng minh model học từ GTSRB **transfer được** sang biển VN thực tế.

### Cách lấy
1. Vào Google Images, search theo từ khóa cột phải bảng dưới
2. Chọn ảnh chụp **rõ nét, biển chiếm > 60% khung**, ánh sáng tốt
3. Mở Paint / Snipping Tool → **crop sát viền biển + chừa ~10% padding 4 cạnh**
4. Save PNG, kích thước sau crop nên trong khoảng **128–256 px**
5. Đặt tên: `vn_01_stop.png`, `vn_02_cam_vao.png`, ...

### Gợi ý 5 ảnh nên lấy (xác suất nhận đúng cao)

| File đề xuất | Từ khóa search Google | GTSRB tương đương | Lý do an toàn |
|---|---|---|---|
| `vn_01_stop.png` | `biển stop dừng lại Việt Nam đường phố` | Stop (14) | Giống hệt biển Đức |
| `vn_02_cam_vao.png` | `biển cấm đi ngược chiều P.102 thực tế` | No entry (17) | Giống hệt |
| `vn_03_cam_vuot.png` | `biển cấm vượt P.125 đường Việt Nam` | No passing (9) | Giống hệt |
| `vn_04_speed_50.png` | `biển hạn chế tốc độ 50 km/h Việt Nam` | Speed 50 (2) | Hình tròn đỏ giống |
| `vn_05_vong_xuyen.png` | `biển vòng xuyến R.303 Việt Nam` | Roundabout (40) | Tròn xanh + mũi tên giống |

### ⚠️ TRÁNH lấy
- Biển **cảnh báo nền vàng** (tam giác vàng VN ≠ tam giác trắng Đức)
- Biển có chữ tiếng Việt (cấm xe gắn máy, đường 1 chiều có chữ...)
- Tốc độ **40 km/h** (GTSRB không có lớp này — sẽ ra 30 hoặc 50)
- Biển chỉ dẫn **hình chữ nhật xanh dương** (GTSRB không có nhóm này)

### Verify trước khi demo
```powershell
python -m src.predict --image demo_images/tier2_vn_real/vn_01_stop.png
```
→ Chấp nhận nếu **top-1 đúng + confidence ≥ 60%**. Nếu fail, đổi ảnh khác.

---

## Tier 3 — `tier3_failure/` 🔧 BẠN CẦN TỰ LẤY (2-3 ảnh)

**Mục đích**: thể hiện em **hiểu giới hạn** của model GTSRB → ăn điểm liêm chính, dẫn dắt qua "future work fine-tune".

### Gợi ý 3 ảnh failure case có giáo dục cao

| File đề xuất | Từ khóa search | Lý do thất bại "đẹp" |
|---|---|---|
| `fail_01_canh_bao_vang.png` | `biển cảnh báo trẻ em qua đường W.225 Việt Nam` | Tam giác **nền vàng** — model train trên nền trắng → predict sai class hoặc confidence thấp |
| `fail_02_cao_toc.png` | `biển báo cao tốc Việt Nam xanh dương` | GTSRB không có biển chỉ dẫn xanh chữ nhật → predict bừa |
| `fail_03_speed_40.png` | `biển hạn chế tốc độ 40 km/h Việt Nam` | GTSRB không có lớp 40 → sẽ predict 30 hoặc 50 với confidence trung bình |

### Cách dùng trong demo
> *"Đây là 3 trường hợp model **không nhận đúng** — em chủ động phân tích để chỉ ra giới hạn:*
> 1. *Biển cảnh báo VN nền vàng vs Đức nền trắng → khác visual feature*
> 2. *Biển chỉ dẫn xanh dương VN không có trong GTSRB → out of distribution*
> 3. *Tốc độ 40 km/h VN không có trong 43 class GTSRB*
>
> *Đây chính là động cơ cho phần **future work**: fine-tune trên dataset VN."*

→ Sau đó mở `notebooks/colab_finetune_vn.ipynb` cho cô xem lộ trình.

---

## Checklist trước demo (D-1 ngày)

- [ ] Tier 1: 10 ảnh đã verify (script trong `DEMO_SCRIPT.md`)
- [ ] Tier 2: ≥ 5 ảnh, mỗi ảnh đã test predict riêng, top-1 đúng + conf ≥ 60%
- [ ] Tier 3: 2-3 ảnh, ghi lại expected behavior (sai gì, confidence bao nhiêu)
- [ ] In giấy A4: 2-3 biển (Stop, Cấm vào) cho phần camera realtime
- [ ] Streamlit chạy được local: `streamlit run app/streamlit_app.py`
- [ ] Notebook fine-tune mở sẵn 1 tab trình duyệt
