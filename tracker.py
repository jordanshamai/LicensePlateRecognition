# tracker.py
import cv2
from utils import calculate_iou
from ocr import most_likely_license_plate

class LicensePlateTracker:
    def __init__(self, iou_threshold=0.3):
        self.tracked_plates = []
        self.iou_threshold = iou_threshold

    def update(self, new_plates, frame, reader):
        new_tracked_plates = []
        for new_plate in new_plates:
            matched = False
            for tracked_plate in self.tracked_plates:
                iou = calculate_iou(new_plate, tracked_plate['box'])
                if iou > self.iou_threshold:
                    tracked_plate['box'] = new_plate
                    xmin, ymin, xmax, ymax = new_plate
                    cropped_plate = frame[ymin:ymax, xmin:xmax]
                    ocr_result = reader.readtext(cropped_plate)
                    if ocr_result:
                        tracked_plate['ocr_results'].extend(ocr_result)
                    new_tracked_plates.append(tracked_plate)
                    matched = True
                    break
            if not matched:
                xmin, ymin, xmax, ymax = new_plate
                cropped_plate = frame[ymin:ymax, xmin:xmax]
                ocr_result = reader.readtext(cropped_plate)
                if ocr_result:
                    new_tracked_plates.append({
                        'box': new_plate,
                        'ocr_results': ocr_result,
                        'frame_count': 0
                    })
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
