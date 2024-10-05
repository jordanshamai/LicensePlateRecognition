# detect.py
import cv2
import torch
from ultralytics import YOLO

CONFIDENCE_THRESHOLD = 0.4
COLOR = (0, 255, 0)

def detect_number_plates(image, model, display=False):
    # Run the YOLO model and get detections (bounding boxes, confidences, and classes)
    detections = model.predict(image)[0].boxes.data
    class_labels = model.names

    if detections.shape == torch.Size([0, 6]):
        print("No objects have been detected.")
        return [], [], []

    all_objects_list = []
    license_plate_list = []
    vehicle_list = []

    for detection in detections:
        xmin, ymin, xmax, ymax = int(detection[0]), int(detection[1]), int(detection[2]), int(detection[3])
        confidence = detection[4]
        class_id = int(detection[5])
        class_label = class_labels[class_id]

        all_objects_list.append({
            'box': [xmin, ymin, xmax, ymax],
            'confidence': confidence,
            'class_label': class_label
        })

        if confidence >= CONFIDENCE_THRESHOLD and class_label.lower() == "license-plate":
            license_plate_list.append([xmin, ymin, xmax, ymax])
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), COLOR, 2)
            text = "License Plate: {:.2f}%".format(confidence * 100)
            cv2.putText(image, text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 2)
            if display:
                number_plate = image[ymin:ymax, xmin:xmax]
                cv2.imshow(f"License plate {len(license_plate_list)}", number_plate)

        if confidence >= CONFIDENCE_THRESHOLD and class_label.lower() == "vehicle":
            vehicle_list.append([xmin, ymin, xmax, ymax])
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            text = "Vehicle: {:.2f}%".format(confidence * 100)
            cv2.putText(image, text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return all_objects_list, license_plate_list, vehicle_list
