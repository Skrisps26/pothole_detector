import time

import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "best_int8_2.tflite"
VIDEO_PATH = "cityRoad_potHoles-side.mp4"

IMG_SIZE = 256
CONF_THRES = 0.15
NMS_THRES = 0.45


def preprocess(frame):
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def postprocess(output, frame_shape):
    h0, w0 = frame_shape
    boxes = []
    scores = []

    output = output[0]  # [5, N]

    for i in range(output.shape[1]):
        cx, cy, w, h = output[0:4, i]
        conf = output[4, i]

        if conf < CONF_THRES:
            continue

        x1 = int((cx - w / 2) * w0)
        y1 = int((cy - h / 2) * h0)
        x2 = int((cx + w / 2) * w0)
        y2 = int((cy + h / 2) * h0)

        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(float(conf))

    return boxes, scores


def main():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Could not open video")
        return

    infer_times = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor = preprocess(frame)

        start = time.time()
        interpreter.set_tensor(input_details[0]["index"], input_tensor)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])
        end = time.time()

        infer_time = end - start
        infer_times.append(infer_time)

        boxes, scores = postprocess(output, frame.shape[:2])

        indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRES, NMS_THRES)

        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(
                    frame,
                    "POTHOLE",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

        fps = 1.0 / infer_time if infer_time > 0 else 0

        cv2.putText(
            frame,
            f"Inference FPS: {fps:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.imshow("TensorFlow Lite Video Benchmark", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    avg = sum(infer_times) / len(infer_times)
    print("\n===== BENCHMARK RESULT =====")
    print(f"Average latency: {avg * 1000:.2f} ms")
    print(f"Average FPS: {1 / avg:.2f}")
    print("============================")


if __name__ == "__main__":
    main()
