# main.py
import cv2
from ultralytics import YOLO
from easyocr import Reader
from detect import detect_number_plates
from ocr import recognize_number_plates
from utils import write_to_csv
from tracker import LicensePlateTracker
import os

def process_image(file_path, model, reader):
    image = cv2.imread(file_path)
    _, number_plate_list, _ = detect_number_plates(image, model, display=True)

    if number_plate_list:
        number_plate_list = recognize_number_plates(file_path, reader, number_plate_list)
        write_to_csv(file_path, number_plate_list)
        for entry in number_plate_list:
            box = entry[0]
            text = entry[1]
            xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
            cv2.putText(image, text, (xmin, ymin - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow('Image', image)
        cv2.waitKey(0)

def process_video(file_path, model, reader):
    video_cap = cv2.VideoCapture(file_path)
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter("output.mp4", fourcc, fps, (frame_width, frame_height))
    plate_tracker = LicensePlateTracker(iou_threshold=0.3)

    while True:
        success, frame = video_cap.read()
        if not success:
            break
        _, number_plate_list, _ = detect_number_plates(frame, model)
        if number_plate_list:
            plate_tracker.update(number_plate_list, frame, reader)
        plate_tracker.draw_plates(frame)
        writer.write(frame)
        cv2.imshow("Output", frame)
        if cv2.waitKey(10) == ord("q"):
            break
    video_cap.release()
    writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model = YOLO("best.pt")
    reader = Reader(['en'], gpu=True)
    file_path = "Videos/Video.mov"
    _, file_extension = os.path.splitext(file_path)

    if file_extension in ['.jpg', '.jpeg', '.png']:
        process_image(file_path, model, reader)
    elif file_extension in ['.mp4', '.mkv', '.avi', '.wmv', '.mov']:
        process_video(file_path, model, reader)
    else:
        print("Unsupported file format.")
