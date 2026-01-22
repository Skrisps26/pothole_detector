import time

import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "best_int8_2.tflite"
IMAGE_PATH = "./pothole.jpg"  # any pothole image
IMG_SIZE = 256
N_RUNS = 200
N_WARMUP = 20

cv2.setNumThreads(4)
cv2.setUseOptimized(True)

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH, num_threads=4)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input dtype:", input_details[0]["dtype"])
print("Input shape:", input_details[0]["shape"])

img = cv2.imread(IMAGE_PATH)
assert img is not None, "Image not found"

img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = img[:, :, ::-1]  # BGR → RGB
img = img.astype(np.float32) / 255.0
img = img[None]  # add batch dimension


for _ in range(N_WARMUP):
    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()

start = time.time()

for _ in range(N_RUNS):
    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()

end = time.time()

total_time = end - start
fps = N_RUNS / total_time
latency_ms = (total_time / N_RUNS) * 1000

print("===== INT8 TFLite Benchmark =====")
print(f"Total runs      : {N_RUNS}")
print(f"Total time (s)  : {total_time:.2f}")
print(f"FPS             : {fps:.2f}")
print(f"Latency (ms)    : {latency_ms:.2f}")
