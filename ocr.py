from paddleocr import PaddleOCR
from collections import defaultdict
import cv2

# Initialize PaddleOCR
def initialize_ocr():
    return PaddleOCR(use_angle_cls=True, lang='en')

# Perform OCR on a cropped image and return detected text
def recognize_number_plates(reader, cropped_path):
    print(f"Performing OCR on {cropped_path}...")
    
    # Load the image
    cropped_image = cv2.imread(cropped_path)
    if cropped_image is None:
        print(f"Error: Unable to load image at {cropped_path}")
        return None, None
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    
    # Apply binarization (Otsu's thresholding)
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Save the preprocessed image (optional for debugging)
    preprocessed_path = f"preprocessed_{cropped_path}"
    cv2.imwrite(preprocessed_path, binary_image)
    
    # Perform OCR on the preprocessed binary image
    ocr_result = reader.ocr(preprocessed_path)

    if ocr_result is None or len(ocr_result) == 0:
        print(f"OCR returned no results for {cropped_path}.")
        return None, None

    # Extract the detected text and confidence
    text = ""
    confidence = None
    if ocr_result[0]:
        for result in ocr_result[0]:
            if result is not None and len(result) > 1:
                detected_text = result[1][0]  # Extract the detected text
                confidence = result[1][1]     # Extract the confidence score
                text += detected_text + " "   # Append to form the full plate text

    # Clean up the detected text
    cleaned_text = filter_text(text.strip())
    if cleaned_text:
        print(f"Detected text: {cleaned_text} (Confidence: {confidence})")
    else:
        print(f"No valid text detected for {cropped_path}.")

    return cleaned_text, confidence

# Function to filter out special characters and handle lowercase letters
def filter_text(text):
    # Characters that look similar to uppercase versions
    similar_letters = {
        'c': 'C', 'o': 'O', 's': 'S', 'v': 'V', 'w': 'W', 'x': 'X', 'z': 'Z',
        'a': 'A', 'b': 'B', 'd': 'D', 'e': 'E', 'g': 'G', 'i': 'I', 'l': 'L', 'm': 'M', 'n': 'N', 'p': 'P', 'q': 'Q', 'r': 'R', 't': 'T', 'u': 'U', 'y': 'Y'
    }

    cleaned_text = []
    for char in text:
        if char.isdigit():
            cleaned_text.append(char)
        elif char.isalpha():
            if char.isupper():
                cleaned_text.append(char)
            elif char in similar_letters:
                cleaned_text.append(similar_letters[char])

    return ''.join(cleaned_text)

# Function to find the most likely license plate text based on OCR results from multiple frames
def most_likely_license_plate(ocr_results):
    char_confidences = defaultdict(lambda: defaultdict(list))

    # Iterate through all OCR results
    for ocr_result in ocr_results:
        if ocr_result and ocr_result[0]:  # Check if ocr_result and the first element exist
            for result in ocr_result:
                # The second element should contain the text and confidence
                for sub_result in result:
                    if isinstance(sub_result, tuple):
                        detected_text = sub_result[0]  # Extract the detected text
                        confidence = sub_result[1]     # Extract the confidence score

                        # Ensure that detected_text is a string
                        if isinstance(detected_text, str):
                            filtered_text = filter_text(detected_text)  # Clean the detected text
                            # Add each character's confidence to the corresponding position
                            for i, char in enumerate(filtered_text):
                                char_confidences[i][char].append(confidence)

    # If no characters were detected, return an empty string
    if not char_confidences:
        return ""

    # Construct the most likely license plate based on average confidence
    most_likely_plate = ""
    for i in sorted(char_confidences.keys()):
        # Find the character with the highest average confidence for each position
        best_char = max(char_confidences[i], key=lambda char: sum(char_confidences[i][char]) / len(char_confidences[i][char]))
        most_likely_plate += best_char

    return most_likely_plate


#Implementing improved logic

# from paddleocr import PaddleOCR
# import cv2
# from collections import defaultdict
# # Initialize PaddleOCR
# def initialize_ocr():
#     return PaddleOCR(use_angle_cls=True, lang='en')

# # Perform OCR on a cropped image and return detected text


# # Perform OCR on a cropped image and return detected text
# def recognize_number_plates(reader, cropped_image):
#     print("Performing OCR...")
    
#     # Convert to grayscale
#     gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    
#     # Apply binarization (Otsu's thresholding)
#     _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
#     # Perform OCR on the preprocessed binary image
#     ocr_result = reader.ocr(binary_image)

#     if ocr_result is None or len(ocr_result) == 0:
#         print("OCR returned no results.")
#         return None, None

#     # Track characters and their confidence across frames
#     char_confidences = defaultdict(list)
#     text = ""
#     confidence = 0

#     # Extract detected text and confidence from the OCR result
#     if ocr_result[0]:
#         for result in ocr_result[0]:
#             detected_text = result[1][0]  # Extract the detected text
#             conf_score = result[1][1]     # Extract the confidence score
#             text += detected_text
#             confidence = conf_score
        
#             # Store each character with its confidence
#             for i, char in enumerate(detected_text):
#                 char_confidences[i].append((char, conf_score))

#     # Combine characters based on their highest confidence from all frames
#     most_likely_plate = combine_characters_by_confidence(char_confidences)

#     # Clean up the detected text
#     cleaned_text = filter_text(most_likely_plate)
#     if cleaned_text:
#         print(f"Detected text: {cleaned_text} (Confidence: {confidence})")
#     else:
#         print(f"No valid text detected.")

#     return cleaned_text, confidence

# # Combine characters from multiple frames based on their confidence
# def combine_characters_by_confidence(char_confidences):
#     most_likely_plate = ""

#     # Iterate over each character position
#     for i in sorted(char_confidences.keys()):
#         # Find the character with the highest confidence for each position
#         best_char = max(char_confidences[i], key=lambda x: x[1])[0]
#         most_likely_plate += best_char

#     return most_likely_plate

# # Function to filter out special characters and handle lowercase letters
# def filter_text(text):
#     # Characters that look similar to uppercase versions
#     similar_letters = {
#         'c': 'C', 'o': 'O', 's': 'S', 'v': 'V', 'w': 'W', 'x': 'X', 'z': 'Z',
#         'a': 'A', 'b': 'B', 'd': 'D', 'e': 'E', 'g': 'G', 'i': 'I', 'l': 'L', 'm': 'M', 'n': 'N', 'p': 'P', 'q': 'Q', 'r': 'R', 't': 'T', 'u': 'U', 'y': 'Y'
#     }

#     cleaned_text = []
#     for char in text:
#         if char.isdigit():
#             cleaned_text.append(char)
#         elif char.isalpha():
#             if char.isupper():
#                 cleaned_text.append(char)
#             elif char in similar_letters:
#                 cleaned_text.append(similar_letters[char])

#     return ''.join(cleaned_text)

