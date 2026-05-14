# 🎥 SPEC — Mở rộng Streamlit app: nhận diện biển báo realtime qua webcam

> Đặc tả kỹ thuật cho tính năng camera realtime, bổ sung vào [`app/streamlit_app.py`](app/streamlit_app.py) hiện tại (đang chỉ hỗ trợ upload ảnh tĩnh). Dùng model `models/custom_cnn_v1.keras` (43 lớp GTSRB) làm classifier, chạy trực tiếp trên frame stream từ webcam của trình duyệt.
>
> **Phạm vi sử dụng**: chỉ chạy local trên máy trình bày để bảo vệ đồ án. Không tính tới deploy lên server / Streamlit Cloud — nên không bàn HTTPS, STUN/TURN, multi-user.

---

## 1. Mục tiêu

- Người dùng mở app trên máy local → bấm "Bật camera" → cấp quyền webcam → app hiển thị live preview kèm nhãn biển báo + confidence trên overlay, cập nhật theo thời gian thực.
- Demo mượt trong buổi bảo vệ: bật được trong < 5 giây, không cần thao tác phức tạp, không phụ thuộc mạng.
- Không phá vỡ luồng upload ảnh tĩnh hiện có.

## 2. Phạm vi

| Có | Không (out of scope giai đoạn này) |
|---|---|
| Live classify 1 ROI ở giữa frame | Detect nhiều biển trong frame |
| Top-k label + confidence overlay | Tracking biển qua nhiều frame |
| Confidence threshold để ẩn kết quả nhiễu | Ghi video output |
| Smoothing nhãn theo N frame gần nhất | Bounding box do user vẽ tự do |
| Snapshot 1 frame để lưu/share | Multi-camera / camera back ↔ front (sẽ có nếu device hỗ trợ qua WebRTC) |

## 3. Lựa chọn công nghệ

| Hạng mục | Giải pháp | Ghi chú |
|---|---|---|
| Stream webcam | **streamlit-webrtc** | Chạy WebRTC trong browser; có sẵn pattern `VideoProcessorBase` |
| Codec frame | **PyAV** (`av`) | Dependency bắt buộc của streamlit-webrtc |
| Inference | **TensorFlow Keras** (model hiện tại) | Tận dụng nguyên `preprocess_single_image` |
| (Tuỳ chọn) tăng tốc | TFLite FP16 (đã xuất ở `models/tflite/`) | Nếu latency CPU > 80 ms/frame |
| Vẽ overlay | **OpenCV** (`cv2.putText`, `cv2.rectangle`) | Đã có trong `requirements.txt` |

Thêm dependency:
```
streamlit-webrtc>=0.47
av>=11.0
```

## 4. Kiến trúc luồng frame

```
Browser webcam ──WebRTC──▶ streamlit-webrtc ──▶ VideoProcessor.recv(frame: av.VideoFrame)
                                                       │
                                                       ▼
                                          frame.to_ndarray(format="bgr24")
                                                       │
                            ┌──────────────────────────┼──────────────────────────┐
                            ▼                          ▼                          ▼
                  Vẽ ROI box (cv2)          Crop ROI center square         (giữ nguyên copy
                  Vẽ label overlay          → resize 48x48                  hiển thị live)
                            │                → /255.0
                            │                → model.predict()
                            │                → top-1 + smoothing
                            └──────────────────────────┘
                                                       │
                                                       ▼
                                       av.VideoFrame.from_ndarray(annotated, "bgr24")
```

## 5. Cấu trúc code (incremental, không phá `streamlit_app.py`)

```
app/
├── streamlit_app.py              # Thêm tab/radio chọn mode: "Upload ảnh" | "Camera realtime"
└── realtime/
    ├── __init__.py
    ├── video_processor.py        # Class SignVideoProcessor(VideoProcessorBase)
    ├── inference.py              # Wrapper load model 1 lần (cache_resource)
    └── overlay.py                # Hàm vẽ ROI box, label, confidence bar
```

`streamlit_app.py` chỉ thêm:
```python
mode = st.sidebar.radio("Chế độ", ["📁 Upload ảnh", "🎥 Camera realtime"])
if mode == "🎥 Camera realtime":
    from app.realtime.video_processor import render_realtime_tab
    render_realtime_tab(model, labels)
else:
    # ... code upload hiện tại giữ nguyên
```

## 6. ROI & preprocessing (PHẢI khớp training)

- ROI = **hình vuông ở giữa frame**, cạnh = `min(h, w) * roi_ratio` (mặc định `roi_ratio=0.5`, có slider 0.3–0.9 trong sidebar).
- Resize bilinear → 48×48, RGB, chia 255.0 → đúng `preprocess_single_image` trong `src/preprocessing.py`.
- **Lý do dùng ROI cố định**: model là classifier 1-biển, không phải detector. Người dùng cần đưa biển vào khung giữa.

## 7. Smoothing & threshold (chống nhãn nhấp nháy)

- Buffer `deque(maxlen=N)` (mặc định N=5) chứa softmax vector của N frame gần nhất.
- Nhãn hiển thị = argmax của **trung bình** N vector.
- Nếu `max(prob_avg) < threshold` (mặc định 0.6, slider 0.3–0.95 trong sidebar) → hiển thị `"— không phát hiện —"` thay vì nhãn rác.

## 8. Performance budget

| Tiêu chí | Ngưỡng | Ghi chú |
|---|---|---|
| Inference / frame (CPU laptop) | < 40 ms | Model nhỏ (333K params) |
| End-to-end FPS hiển thị | ≥ 12 FPS | streamlit-webrtc tự throttle |
| Cold start (load model) | < 2 s | Dùng `@st.cache_resource` |
| Băng thông WebRTC | < 1 Mbps | Đặt `media_stream_constraints={"video": {"width": 640}}` |

Nếu latency cao: chuyển sang **TFLite FP16** (`models/tflite/custom_cnn_v1_fp16.tflite`) qua `tf.lite.Interpreter` — dùng cùng API trong `inference.py`.

## 9. UX sidebar (mode realtime)

- Slider `ROI ratio` (0.3–0.9, default 0.5).
- Slider `Confidence threshold` (0.30–0.95, default 0.60).
- Slider `Smoothing window` (1–10, default 5).
- Toggle `Hiển thị top-3` (default off — chỉ top-1 trên overlay).
- Nút `📸 Chụp snapshot` → lưu PNG vào `reports/snapshots/YYYYMMDD_HHMMSS.png` kèm nhãn dự đoán.
- Badge trạng thái: "🟢 Đang chạy / 🟡 Đang load model / 🔴 Không có camera".

## 10. Chạy local

- Khởi động: `streamlit run app/streamlit_app.py` → mở `http://localhost:8501`.
- Trình duyệt cho phép `getUserMedia` trên `localhost` mà không cần HTTPS → bật camera được ngay.
- Toàn bộ inference chạy trong process Streamlit của máy local; không gửi frame ra ngoài.
- Trước khi bảo vệ: chạy thử 1 lần ở chính máy + camera sẽ dùng để chắc chắn driver/quyền OK.

## 11. Testing

- **Unit**: `tests/test_overlay.py` — kiểm tra hàm vẽ overlay không crash với ảnh đen / size khác nhau.
- **Unit**: `tests/test_video_processor.py` — feed `np.zeros((480,640,3), uint8)` qua `recv()`, assert output cùng shape & là `av.VideoFrame`.
- **Smoke (dùng cho rehearsal bảo vệ)**: chạy `streamlit run` local, mở `localhost:8501`, bật camera, đưa 5 ảnh biển báo từ `data/processed/test/` lên màn hình → verify đúng nhãn xuất hiện.
- **Performance**: log FPS thực tế trong `VideoProcessor` (counter `frames / elapsed`), in lên overlay góc dưới-phải khi bật toggle `Show debug`.

## 12. Roadmap triển khai

| Bước | Việc | Output |
|---|---|---|
| 1 | Cài `streamlit-webrtc` + `av`, cập nhật `requirements.txt` | requirements xanh |
| 2 | Tạo `app/realtime/inference.py` — wrap `load_model_and_labels()` để dùng chung 2 mode | Module test pass |
| 3 | Tạo `overlay.py` (vẽ ROI box + label) | Unit test pass |
| 4 | Tạo `video_processor.py` với `SignVideoProcessor` (ROI + smoothing + threshold) | Smoke test bật camera thấy overlay |
| 5 | Wire vào `streamlit_app.py` qua `st.sidebar.radio` | Cả 2 mode hoạt động |
| 6 | Snapshot button + lưu file | File ra đúng `reports/snapshots/` |
| 7 | (Tuỳ chọn) backend TFLite FP16 | Latency < 25 ms/frame |
| 8 | Cập nhật README mục "Bước 5 — Demo web" | Tài liệu đồng bộ |

## 13. Rủi ro & mitigation

| Rủi ro | Mitigation |
|---|---|
| Máy bảo vệ không có camera / driver lỗi | Test trước trên đúng máy + camera; chuẩn bị sẵn 1 video file MP4 chứa các biển báo làm phương án backup |
| Model không nhận diện được biển VN ngoài 43 lớp GTSRB | Hiển thị banner "Bộ biển GTSRB (Đức)"; thêm threshold để giảm false positive |
| Frame BGR vs RGB lẫn lộn → ảnh sai màu vào model | Test parity: chạy cùng ảnh qua upload mode và realtime mode, prediction phải giống |
| Tab sleep / browser throttle giảm FPS | streamlit-webrtc tự xử; thông báo user nếu FPS < 5 |
| TF model load chậm lần đầu (>3s) | Hiển thị spinner; cache_resource đảm bảo chỉ load 1 lần |

## 14. Phụ lục — skeleton `SignVideoProcessor`

```python
import av
import cv2
import numpy as np
from collections import deque
from streamlit_webrtc import VideoProcessorBase
from src import config as C
from src.preprocessing import preprocess_single_image
import tensorflow as tf

class SignVideoProcessor(VideoProcessorBase):
    def __init__(self, model, labels, roi_ratio=0.5, threshold=0.6, smooth=5):
        self.model, self.labels = model, labels
        self.roi_ratio, self.threshold = roi_ratio, threshold
        self.buf = deque(maxlen=smooth)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        side = int(min(h, w) * self.roi_ratio)
        x0, y0 = (w - side) // 2, (h - side) // 2
        roi = cv2.cvtColor(img[y0:y0+side, x0:x0+side], cv2.COLOR_BGR2RGB)
        x = preprocess_single_image(tf.convert_to_tensor(roi), C.IMG_SIZE)[None, ...]
        probs = self.model.predict(x, verbose=0)[0]
        self.buf.append(probs)
        avg = np.mean(self.buf, axis=0)
        idx, conf = int(np.argmax(avg)), float(np.max(avg))
        label = self.labels[idx] if conf >= self.threshold else "— không phát hiện —"
        cv2.rectangle(img, (x0, y0), (x0+side, y0+side), (0, 255, 0), 2)
        cv2.putText(img, f"{label}  {conf*100:.1f}%", (x0, y0-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")
```
