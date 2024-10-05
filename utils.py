# utils.py
import csv
import os

CSV_FILE_PATH = "number_plates.csv"

def write_to_csv(file_path, number_plate_list):
    with open(CSV_FILE_PATH, mode='a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        if os.stat(CSV_FILE_PATH).st_size == 0:
            csv_writer.writerow(["image_path", "box", "cleaned_text"])

        for plate_info in number_plate_list:
            box = plate_info[0]
            cleaned_text = plate_info[1]
            csv_writer.writerow([file_path, box, cleaned_text])

def calculate_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    iou = interArea / float(box1Area + box2Area - interArea)
    return iou
