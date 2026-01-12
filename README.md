# Road Anomaly / Pothole Detection (Edge AI – Raspberry Pi)

Real-time **road anomaly (pothole) detection** using a **YOLO11n INT8 TFLite model** running fully on **CPU**.  
Built for **edge deployment** on **Raspberry Pi 4B**, also works on a regular PC.

Made for Raspberry Pi 4B, use webcam or camera module with it.

Can be used on other devices, even windows but why would you do that?

---

## 🔥 Key Features

- 🚀 Real-time inference on CPU (INT8 TFLite)
- 📷 Live webcam input (USB camera / laptop camera)
- 🧠 Single-class detection (`pothole`)
- ⚡ Optimized for Raspberry Pi 4B
- 🌐 Fully edge-based (no internet, no cloud)

---

## 🧠 Model Details

- **Model**: YOLO11n (Ultralytics)
- **Task**: Object Detection (single class)
- **Training resolution**: 640×640
- **Inference resolution**: 320×320
- **Model format**: TFLite (INT8)
- **Output format**:Coordinates+confidence

## Execution

```bash
pip install tensorflow numpy opencv-python
python potdetector.py

```


uwu
