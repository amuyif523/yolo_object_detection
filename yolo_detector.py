import cv2
import numpy as np
import os
import time
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=False,
                help="path to input video file; leave blank to use webcam")
ap.add_argument("-o", "--output", required=False,
                help="path to output video file to save the result")
args = vars(ap.parse_args())

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
    (H, W) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(output_layers)

    boxes, confidences, classIDs = [], [], []

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
            (x, y, w, h) = (boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3])
            color = [int(c) for c in colors[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = f"{classes[classIDs[i]]}: {confidences[i]:.4f}"
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

if __name__ == '__main__':
    input_source = args["input"] if args["input"] else 0
    print(f"[INFO] Starting video stream from: {'Webcam' if input_source == 0 else input_source}...")
    vs = cv2.VideoCapture(input_source)
    
    if not vs.isOpened():
        print(f"[ERROR] Could not open video source: {input_source}")
        exit()

    writer = None
    (W, H) = (None, None)

    while True:
        (grabbed, frame) = vs.read()
        if not grabbed:
            print("[INFO] End of video stream reached.")
            break

        if W is None or H is None:
            (H, W) = frame.shape[:2]

        start_time = time.time()
        processed_frame = process_frame(frame)
        end_time = time.time()
        
        fps = 1 / (end_time - start_time)
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(processed_frame, fps_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Real-Time Object Detection", processed_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        if args["output"] and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"], fourcc, 30, (W, H), True)
        
        if writer is not None:
            writer.write(processed_frame)

    print("[INFO] Cleaning up and closing windows...")
    vs.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

