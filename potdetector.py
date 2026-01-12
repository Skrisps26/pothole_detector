import time

import cv2
import numpy as np
import tensorflow as tf

# ---------------- CONFIG ----------------
MODEL_PATH = "best_int8.tflite"
IMG_SIZE = 320
CONF_THRES = 0.4
NMS_THRES = 0.4
CAMERA_INDEX = 0  # 0 for PC webcam, usually 0 for Pi USB cam
# ----------------------------------------

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH, num_threads=4)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Model loaded")
print("Input:", input_details[0]["shape"])
print("Output:", output_details[0]["shape"])

# Open webcam
cap = cv2.VideoCapture(CAMERA_INDEX)
assert cap.isOpened(), "Webcam not detected"

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h0, w0 = frame.shape[:2]

    # ---------- Preprocess ----------
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = img[:, :, ::-1].astype(np.float32) / 255.0
    img = img[None]

    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()

    # ---------- Output decode ----------
    output = interpreter.get_tensor(output_details[0]["index"])
    output = output[0].transpose(1, 0)  # (2100, 5)

    boxes = []
    scores = []

    sx = w0 / IMG_SIZE
    sy = h0 / IMG_SIZE

    for det in output:
        cx, cy, w, h, conf = det

        if conf < CONF_THRES:
            continue

        # area filter (kills cracks)
        if w * h < 0.01:
            continue

        # normalized → model pixels
        cx *= IMG_SIZE
        cy *= IMG_SIZE
        w *= IMG_SIZE
        h *= IMG_SIZE

        # center → corner
        x1 = (cx - w / 2) * sx
        y1 = (cy - h / 2) * sy
        x2 = (cx + w / 2) * sx
        y2 = (cy + h / 2) * sy

        x1 = int(max(0, min(x1, w0 - 1)))
        y1 = int(max(0, min(y1, h0 - 1)))
        x2 = int(max(0, min(x2, w0 - 1)))
        y2 = int(max(0, min(y2, h0 - 1)))

        bw = x2 - x1
        bh = y2 - y1
        if bw <= 5 or bh <= 5:
            continue

        boxes.append([x1, y1, bw, bh])
        scores.append(float(conf))

    # ---------- NMS ----------
    indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRES, NMS_THRES)

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Pothole {scores[i]:.2f}",
                (x, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

    # ---------- FPS ----------
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )

    cv2.imshow("Road Anomaly Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
