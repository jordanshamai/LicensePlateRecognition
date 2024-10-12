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
                cv2.imwrite(f"cropped_plate.jpg", number_plate)

        if confidence >= CONFIDENCE_THRESHOLD and class_label.lower() == "vehicle":
            vehicle_list.append([xmin, ymin, xmax, ymax])
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            text = "Vehicle: {:.2f}%".format(confidence * 100)
            cv2.putText(image, text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return all_objects_list, license_plate_list, vehicle_list


# Implementing better YOLO logic
# import cv2
# import torch
# from ultralytics import YOLO
# from ocr import recognize_number_plates  # Import the OCR function from ocr.py

# # Set confidence thresholds for YOLO and OCR
# YOLO_CONFIDENCE_THRESHOLD = 0.6
# OCR_CONFIDENCE_THRESHOLD = 0.8
# COLOR = (0, 255, 0)

# def detect_number_plates(image, model, ocr_reader, display=False):
#     """
#     Detect license plates and vehicles, and run OCR on license plates with a high confidence threshold.
#     The YOLO detection model only runs again if the OCR confidence is low.
#     """
#     while True:
#         # Run the YOLO model and get detections (bounding boxes, confidences, and classes)
#         detections = model.predict(image)[0].boxes.data
#         class_labels = model.names

#         if detections.shape == torch.Size([0, 6]):
#             print("No objects have been detected.")
#             return [], [], []

#         all_objects_list = []
#         license_plate_list = []
#         vehicle_list = []
#         rerun_detection = False

#         for detection in detections:
#             xmin, ymin, xmax, ymax = int(detection[0]), int(detection[1]), int(detection[2]), int(detection[3])
#             confidence = detection[4]
#             class_id = int(detection[5])
#             class_label = class_labels[class_id]

#             all_objects_list.append({
#                 'box': [xmin, ymin, xmax, ymax],
#                 'confidence': confidence,
#                 'class_label': class_label
#             })

#             # Detect and handle license plates
#             if confidence >= YOLO_CONFIDENCE_THRESHOLD and class_label.lower() == "license-plate":
#                 license_plate_list.append([xmin, ymin, xmax, ymax])
#                 cropped_plate = image[ymin:ymax, xmin:xmax]
#                 cropped_path = f"cropped_plate_{len(license_plate_list)}.jpg"
#                 cv2.imwrite(cropped_path, cropped_plate)

#                 # Run OCR on the detected plate
#                 text, ocr_confidence = recognize_number_plates(ocr_reader, cropped_path)
#                 if text is not None:
#                     print(f"OCR Text: {text}, Confidence: {ocr_confidence}")

#                     if ocr_confidence >= OCR_CONFIDENCE_THRESHOLD:
#                         # High OCR confidence, no need to rerun detection
#                         print(f"High OCR confidence ({ocr_confidence}), continuing detection.")
#                         # Draw bounding box and text
#                         cv2.rectangle(image, (xmin, ymin), (xmax, ymax), COLOR, 2)
#                         cv2.putText(image, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR, 2)
#                     else:
#                         print(f"Low OCR confidence ({ocr_confidence}), rerunning detection...")
#                         rerun_detection = True  # Rerun detection if OCR confidence is too low

#                 else:
#                     print("No OCR result, rerunning detection...")
#                     rerun_detection = True

#                 if display:
#                     cv2.imshow(f"License plate {len(license_plate_list)}", cropped_plate)

#             # Detect and handle vehicles
#             if confidence >= YOLO_CONFIDENCE_THRESHOLD and class_label.lower() == "vehicle":
#                 vehicle_list.append([xmin, ymin, xmax, ymax])
#                 cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
#                 text = "Vehicle: {:.2f}%".format(confidence * 100)
#                 cv2.putText(image, text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#         # If we need to rerun detection, continue the loop; otherwise, break
#         if not rerun_detection:
#             break

#     return all_objects_list, license_plate_list, vehicle_list

