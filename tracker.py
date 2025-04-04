import cv2
from utils import calculate_iou
from ocr import most_likely_license_plate

class LicensePlateTracker:
    def __init__(self, iou_threshold=0.3, max_disappeared=3):
        self.tracked_plates = []
        self.iou_threshold = iou_threshold
        self.max_disappeared = max_disappeared  # Maximum frames a plate can disappear before being removed

    def update(self, new_plates, frame, reader):
        new_tracked_plates = []
        for new_plate in new_plates:
            matched = False
            for tracked_plate in self.tracked_plates:
                iou = calculate_iou(new_plate, tracked_plate['box'])
                if iou > self.iou_threshold:
                    tracked_plate['box'] = new_plate
                    tracked_plate['frame_count'] = 0  # Reset the frame count if the plate is detected again
                    xmin, ymin, xmax, ymax = new_plate
                    cropped_plate = frame[ymin:ymax, xmin:xmax]

                    # Perform OCR on the cropped plate
                    ocr_result = reader.ocr(cropped_plate)
                    if ocr_result and len(ocr_result) > 0:
                        tracked_plate['ocr_results'].extend(ocr_result)
                    new_tracked_plates.append(tracked_plate)
                    matched = True
                    break

            # Add new plate if it wasn't matched with existing tracked plates
            if not matched:
                xmin, ymin, xmax, ymax = new_plate
                cropped_plate = frame[ymin:ymax, xmin:xmax]

                # Perform OCR on the cropped plate
                ocr_result = reader.ocr(cropped_plate)
                if ocr_result and len(ocr_result) > 0:
                    new_tracked_plates.append({
                        'box': new_plate,
                        'ocr_results': [ocr_result],
                        'frame_count': 0
                    })

        # Increment frame count for plates that were not updated (not detected in the current frame)
        for tracked_plate in self.tracked_plates:
            if tracked_plate not in new_tracked_plates:
                tracked_plate['frame_count'] += 1
                # Only keep the plate if it hasn't disappeared for too long
                if tracked_plate['frame_count'] <= self.max_disappeared:
                    new_tracked_plates.append(tracked_plate)

        # Update the tracked plates
        self.tracked_plates = new_tracked_plates

    def draw_plates(self, frame):
        for plate in self.tracked_plates:
            box = plate['box']
            ocr_results = plate['ocr_results']
            most_likely_plate = most_likely_license_plate(ocr_results)
            xmin, ymin, xmax, ymax = box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            hot_pink = (255, 105, 180)
            cv2.putText(frame, most_likely_plate, (xmin, ymax + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, hot_pink, 2)
# 
# 
# 
# improving logic
# 
# import cv2
# from utils import calculate_iou
# from ocr import recognize_number_plates

# class LicensePlateTracker:
#     def __init__(self, iou_threshold=0.3, max_disappeared=3):
#         self.tracked_plates = []
#         self.iou_threshold = iou_threshold
#         self.max_disappeared = max_disappeared  # Maximum frames a plate can disappear before being removed

#     def update(self, new_plates, frame, reader):
#         new_tracked_plates = []
#         for new_plate in new_plates:
#             matched = False
#             for tracked_plate in self.tracked_plates:
#                 iou = calculate_iou(new_plate, tracked_plate['box'])
#                 if iou > self.iou_threshold:
#                     tracked_plate['box'] = new_plate
#                     tracked_plate['frame_count'] = 0  # Reset the frame count if the plate is detected again
#                     xmin, ymin, xmax, ymax = new_plate
#                     cropped_plate = frame[ymin:ymax, xmin:xmax]

#                     # Perform OCR on the cropped plate and accumulate results
#                     cleaned_text, confidence = recognize_number_plates(reader, cropped_plate)
#                     if cleaned_text:
#                         tracked_plate['ocr_results'].append((cleaned_text, confidence))

#                     new_tracked_plates.append(tracked_plate)
#                     matched = True
#                     break

#             # Add new plate if it wasn't matched with existing tracked plates
#             if not matched:
#                 xmin, ymin, xmax, ymax = new_plate
#                 cropped_plate = frame[ymin:ymax, xmin:xmax]

#                 # Perform OCR on the cropped plate and store results
#                 cleaned_text, confidence = recognize_number_plates(reader, cropped_plate)
#                 if cleaned_text:
#                     new_tracked_plates.append({
#                         'box': new_plate,
#                         'ocr_results': [(cleaned_text, confidence)],
#                         'frame_count': 0
#                     })

#         # Increment frame count for plates that were not updated (not detected in the current frame)
#         for tracked_plate in self.tracked_plates:
#             if tracked_plate not in new_tracked_plates:
#                 tracked_plate['frame_count'] += 1
#                 # Only keep the plate if it hasn't disappeared for too long
#                 if tracked_plate['frame_count'] <= self.max_disappeared:
#                     new_tracked_plates.append(tracked_plate)

#         # Update the tracked plates
#         self.tracked_plates = new_tracked_plates

#     def draw_plates(self, frame):
#         for plate in self.tracked_plates:
#             box = plate['box']
#             ocr_results = plate['ocr_results']

#             # Get the most confident OCR result from the tracked results
#             most_confident_result = max(ocr_results, key=lambda x: x[1], default=("", 0))[0]
            
#             xmin, ymin, xmax, ymax = box
#             cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
#             hot_pink = (255, 105, 180)
#             cv2.putText(frame, most_confident_result, (xmin, ymax + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, hot_pink, 2)
