import cv2
from collections import defaultdict

def segment_characters(np_image):
    """
    Segments the characters from a license plate image using OpenCV.
    
    Args:
    - np_image: The cropped license plate image.
    
    Returns:
    - A list of bounding boxes for each detected character.
    """
    gray = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
    
    # Apply binary thresholding to separate characters from background
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours which should represent individual characters
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    character_bboxes = []
    
    # Iterate over contours and filter based on size
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h > 0.5 * np_image.shape[0]:  # Adjust threshold to filter out small noise
            character_bboxes.append([x, y, x + w, y + h])
    
    # Sort character bounding boxes from left to right
    character_bboxes = sorted(character_bboxes, key=lambda bbox: bbox[0])
    
    return character_bboxes

def recognize_number_plates(image_or_path, reader, number_plate_list):
    image = cv2.imread(image_or_path) if isinstance(image_or_path, str) else image_or_path

    for i, box in enumerate(number_plate_list):
        xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
        np_image = image[ymin:ymax, xmin:xmax]

        # Segment characters from the license plate image
        character_bboxes = segment_characters(np_image)

        plate_text = ""
        
        # Perform OCR on each segmented character
        for char_bbox in character_bboxes:
            x1, y1, x2, y2 = char_bbox
            char_image = np_image[y1:y2, x1:x2]
            
            detection = reader.readtext(char_image)
            if detection:
                plate_text += detection[0][1]  # Append the detected character to plate text
        
        # Clean the text and update the number plate list
        cleaned_text = ''.join([char for char in plate_text if char.isalnum()])
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
