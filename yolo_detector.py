import cv2
import numpy as np
import os

weights_path = os.path.join("yolo_model", "yolov3.weights")
config_path = os.path.join("yolo_model", "yolov3.cfg")
names_path = os.path.join("yolo_model", "coco.names")

print("[INFO] Loading YOLO model from disk...")
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

try:
    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    print(f"[ERROR] The file {names_path} was not found. Please ensure it is in the 'yolo_model' directory.")
    exit()

layer_names = net.getLayerNames()

output_layer_indices = net.getUnconnectedOutLayers()

if isinstance(output_layer_indices, np.ndarray) and output_layer_indices.ndim > 1:
    output_layer_indices = output_layer_indices.flatten()


output_layers = [layer_names[i - 1] for i in output_layer_indices]

print("[INFO] YOLO model loaded successfully.")
print(f"[INFO] Model can detect the following {len(classes)} classes:")
print(classes)

if __name__ == '__main__':
    print("\n[SUCCESS] Phase 2 script executed without errors. Model is ready.")

