import cv2
import numpy as np
import os
import time

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

np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")

layer_names = net.getLayerNames()
output_layer_indices = net.getUnconnectedOutLayers() 

if isinstance(output_layer_indices, np.ndarray) and output_layer_indices.ndim > 1:
    output_layer_indices = output_layer_indices.flatten()
    
output_layers = [layer_names[i - 1] for i in output_layer_indices]

def process_frame(frame):
    """
    Takes a video frame, performs object detection, and returns the frame with detections drawn on it.
    """
    (H, W) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
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

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in colors[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = f"{classes[classIDs[i]]}: {confidences[i]:.4f}"
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
    return frame

if __name__ == '__main__':
    print("[INFO] Starting video stream...")
    vs = cv2.VideoCapture(0)

    if not vs.isOpened():
        print("[ERROR] Could not open webcam. Please check if it's connected and not in use by another application.")
        exit()

    time.sleep(2.0) 
    while True:
        (grabbed, frame) = vs.read()

        if not grabbed:
            print("[INFO] End of video stream reached.")
            break

        processed_frame = process_frame(frame)

        cv2.imshow("Real-Time Object Detection", processed_frame)

        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    print("[INFO] Cleaning up and closing windows...")
    vs.release()
    cv2.destroyAllWindows()

