# ocr.py
import cv2
from collections import defaultdict
def recognize_number_plates(image_or_path, reader, number_plate_list):
    image = cv2.imread(image_or_path) if isinstance(image_or_path, str) else image_or_path

    for i, box in enumerate(number_plate_list):
        xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
        np_image = image[ymin:ymax, xmin:xmax]

        detection = reader.readtext(np_image)
        text = detection[0][1] if detection else ""
        cleaned_text = ''.join([char for char in text if char.isalnum()])
        number_plate_list[i] = [box, cleaned_text]

    return number_plate_list

def filter_and_clean_text(text):
    similar_letters = {'c', 'o', 's', 'v', 'w', 'x', 'z'}
    cleaned_text = []
    for char in text:
        if char.isdigit():
            cleaned_text.append(char)
        elif char.isalpha():
            if char.isupper():
                cleaned_text.append(char)
            elif char in similar_letters:
                cleaned_text.append(char.upper())
    return ''.join(cleaned_text)

def most_likely_license_plate(ocr_results):
    position_confidences = collect_character_confidences(ocr_results)
    return construct_most_likely_plate(position_confidences)

def collect_character_confidences(ocr_results):
    position_confidences = defaultdict(lambda: defaultdict(list))
    for result in ocr_results:
        text = result[1]
        cleaned_text = filter_and_clean_text(text)
        if len(cleaned_text) >= 6:
            for i in range(6):
                char = cleaned_text[i]
                position_confidences[i][char].append(result[2])
    return position_confidences

def construct_most_likely_plate(position_confidences):
    most_likely_plate = ""
    for pos in range(6):
        if pos in position_confidences:
            most_likely_char = max(position_confidences[pos], key=lambda x: sum(position_confidences[pos][x]) / len(position_confidences[pos][x]))
            most_likely_plate += most_likely_char
    return most_likely_plate
