import cv2
import numpy as np
import os

CONFIDENCE_THRESHOLD = 0.5

NMS_THRESHOLD = 0.4

weights_path = os.path.join("yolo_model", "yolov3.weights")
config_path = os.path.join("yolo_model", "yolov3.cfg")
names_path = os.path.join("yolo_model", "coco.names")

print("[INFO] Loading YOLO model from disk...")
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

try:
    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    print(f"[ERROR] The file {names_path} was not found.")
    exit()

layer_names = net.getLayerNames()
output_layer_indices = net.getUnconnectedOutLayers()
if isinstance(output_layer_indices, np.ndarray) and output_layer_indices.ndim > 1:
    output_layer_indices = output_layer_indices.flatten()
output_layers = [layer_names[i - 1] for i in output_layer_indices]

image_path = os.path.join("images", "test_image.jpg")
try:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    (H, W) = image.shape[:2]
except Exception as e:
    print(f"[ERROR] Could not read image from {image_path}. Make sure it exists and is a valid image file.")
    print(e)
    exit()

blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

print("[INFO] Performing forward pass and getting detections...")
net.setInput(blob)
layerOutputs = net.forward(output_layers)

boxes = []
confidences = []
classIDs = []

for output in layerOutputs:

    for detection in output:
        scores = detection[5:]      
        classID = np.argmax(scores) 
        confidence = scores[classID]

        if confidence > CONFIDENCE_THRESHOLD:

            box = detection[0:4] * np.array([W, H, W, H])
            (centerX, centerY, width, height) = box.astype("int")

            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))

            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)

print(f"[INFO] Found {len(boxes)} raw boxes. Applying NMS...")
indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
print(f"[INFO] NMS kept {len(indices) if isinstance(indices, np.ndarray) else 0} boxes.")

if len(indices) > 0:
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")

    for i in indices.flatten():
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])

        color = [int(c) for c in colors[classIDs[i]]]

        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

        text = f"{classes[classIDs[i]]}: {confidences[i]:.4f}"

        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

cv2.imshow("Object Detection Result", image)

output_path = os.path.join("images", "output_image.jpg")
cv2.imwrite(output_path, image)
print(f"[INFO] Saved output image to {output_path}")

cv2.waitKey(0)
cv2.destroyAllWindows()

