# 🎥 Nhật ký phát triển — Camera Realtime Mode

Tài liệu ghi lại quá trình implement mode "Camera realtime" cho Streamlit app, bám theo spec ở [`REALTIME_CAMERA_SPEC.md`](REALTIME_CAMERA_SPEC.md).

## 1. Mục tiêu

- Bổ sung tab **🎥 Camera realtime** vào Streamlit app hiện có (giữ nguyên 100% luồng upload cũ).
- Sử dụng `streamlit-webrtc` để stream webcam qua trình duyệt → predict mỗi frame → vẽ ROI + nhãn.
- Smoothing nhãn để chống nhấp nháy; có ngưỡng confidence để loại nhiễu.
- Cho phép chụp snapshot lưu ra `reports/snapshots/`.

## 2. Roadmap đã thực hiện (B1 → B8)

| # | Việc | File / Lệnh | Trạng thái |
|---|---|---|---|
| B1 | Cài `streamlit-webrtc 0.65.4` + `av 16.1.0` + `tornado 6.5.5` | `requirements.txt` | ✅ |
| B2 | Wrapper inference dùng chung 2 mode | `app/realtime/inference.py` (31 LOC) | ✅ |
| B3 | Vẽ ROI box + nhãn lên frame BGR | `app/realtime/overlay.py` (35 LOC) | ✅ |
| B4 | VideoProcessor (worker thread WebRTC) | `app/realtime/video_processor.py` (80 LOC) | ✅ |
| B5 | Sidebar radio chọn 2 mode | `app/streamlit_app.py` (refactor) | ✅ |
| B6 | Nút 📸 chụp snapshot | `app/realtime/tab.py` (76 LOC) | ✅ |
| B7 | Smoke test (import + HTTP) | — | ✅ |
| B8 | README cập nhật mô tả 2 mode | `README.md` Bước 5 | ✅ |

## 3. Cấu trúc file mới

```
app/
├── streamlit_app.py        # Entry point (refactor: thêm sidebar.radio)
└── realtime/               # Mới
    ├── __init__.py
    ├── inference.py        # load_model_and_labels, predict_array, predict_proba
    ├── overlay.py          # compute_center_roi, draw_overlay (cv2 + BGR)
    ├── video_processor.py  # SignVideoProcessor(VideoProcessorBase)
    └── tab.py              # render_realtime_tab + _save_snapshot

reports/
└── snapshots/              # Tự tạo khi nhấn 📸 (gitignore khuyến nghị)
```

## 4. Quyết định kỹ thuật

| Vấn đề | Cách giải quyết |
|---|---|
| Inference latency trên Keras | Dùng full Keras (FP32) trước; nếu < 10 FPS sẽ thay bằng TFLite interpreter (mục 9 spec — chưa làm) |
| Predict block UI | `streamlit-webrtc` chạy `recv()` trong worker thread riêng; main thread chỉ render UI |
| Nhãn nhấp nháy | Smoothing rolling-mean softmax với `deque(maxlen=N)`, default N=5 |
| Race condition khi snapshot | `threading.Lock` bảo vệ `_last_frame_bgr` / `_last_label` / `_last_conf` |
| Thay tham số sau khi started | `ctx.video_processor.configure(...)` mỗi rerun (Streamlit rerun trên mỗi slider change) |
| Model là classifier, không phải detector | ROI center crop cố định + cảnh báo trong UI ("Đưa biển vào khung vuông xanh") |

## 5. Smoke test (2026-05-14)

Môi trường: Windows, Python 3.13.7, TensorFlow 2.21.0, Streamlit 1.57.0, streamlit-webrtc 0.65.4.

```
>>> Import check
All imports OK

>>> streamlit run app/streamlit_app.py --server.port 8765 --server.headless true
Uvicorn server started on 0.0.0.0:8765
GET /_stcore/health  →  200  "ok"
GET /                →  200  (5381 bytes)
```

Không test thật camera vì cần user grant permission trên trình duyệt.

## 6. Sự cố gặp phải & cách xử lý

| Lỗi | Nguyên nhân | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'tornado'` khi import `streamlit_webrtc.eventloop` | Streamlit 1.57 đã drop `tornado` khỏi direct deps (chuyển sang ASGI/uvicorn), nhưng `streamlit-webrtc 0.65` vẫn dùng `tornado.platform.asyncio` | `pip install tornado` + thêm `tornado>=6.4` vào `requirements.txt` với comment giải thích |
| PowerShell `Tee-Object` báo `NativeCommandError` khi `pip install` | pip ghi notice ra stderr → PowerShell hiểu nhầm là exception | Chuyển sang `python -m pip install ...` thuần, kiểm tra return code thay vì stderr |
| Streamlit warning "missing ScriptRunContext" khi import-check ngoài runtime | Bình thường khi gọi cache decorators ngoài `streamlit run` | Bỏ qua; chỉ chạy ở smoke test |

## 7. Cách dùng

```bash
streamlit run app/streamlit_app.py
```

Trong app:
1. Sidebar → chọn **🎥 Camera realtime**.
2. Bấm **START** trên khung video, cho phép trình duyệt dùng camera.
3. Đưa biển báo vào khung vuông xanh ở giữa.
4. Tham số khuyến nghị: `ROI 0.4–0.6`, `threshold 0.6–0.75`, `smoothing 5–7`.
5. Bấm **📸 Chụp snapshot** → file PNG lưu tại `reports/snapshots/<timestamp>_<label>_<conf>.png`.

## 8. Phần spec chưa làm (nice-to-have)

- **TFLite backend** (spec mục 9): swap `predict_proba()` sang `tf.lite.Interpreter` để tăng FPS trên CPU.
- **Multi-camera select**: hiện UA/trình duyệt tự chọn; chưa có dropdown.
- **HTTPS / TURN server**: bỏ qua vì spec đã chốt "chỉ chạy local trong defense".
- **Pytest cho overlay & inference**: chưa viết test (`tests/test_realtime.py`).

## 9. Tham khảo

- Spec: [`REALTIME_CAMERA_SPEC.md`](REALTIME_CAMERA_SPEC.md)
- Preprocessing parity: [`PREPROCESSING.md`](PREPROCESSING.md), `src/preprocessing.py::preprocess_single_image`
- streamlit-webrtc docs: <https://github.com/whitphx/streamlit-webrtc>
- Báo cáo TFLite (cùng codebase, dùng cho repo Android riêng): [`EXPORT_TFLITE.md`](EXPORT_TFLITE.md)
