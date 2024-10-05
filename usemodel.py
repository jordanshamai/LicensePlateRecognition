from ultralytics import YOLO
from easyocr import Reader
import time
import torch
import cv2
import os
import csv
import numpy

from collections import defaultdict, Counter
# Constants
CONFIDENCE_THRESHOLD = 0.4
COLOR = (0, 255, 0)
CSV_FILE_PATH = "number_plates.csv"


def detect_number_plates(image, model, display=False):
    start = time.time()

    # Run the YOLO model and get detections (bounding boxes, confidences, and classes)
    detections = model.predict(image)[0].boxes.data  # Assumes this returns [x1, y1, x2, y2, confidence, class_id]
    class_labels = model.names  # Get class labels from the YOLO model
    
    # Check if there are any detections
    if detections.shape == torch.Size([0, 6]):
        print("No objects have been detected.")
        return [], [], []  # Return three empty lists

    # Initialize the list of all bounding boxes, confidences, and class IDs
    all_objects_list = []
    license_plate_list = []
    vehicle_list = []  # List to hold detected vehicles
    
    # Loop over detections
    for detection in detections:
        xmin, ymin, xmax, ymax = int(detection[0]), int(detection[1]), int(detection[2]), int(detection[3])
        confidence = detection[4]
        class_id = int(detection[5])
        class_label = class_labels[class_id]  # Get the class label for the detected object
        
        # Append detection to the all_objects_list
        all_objects_list.append({
            'box': [xmin, ymin, xmax, ymax],
            'confidence': confidence,
            'class_label': class_label
        })
        
        # Filter detections by confidence threshold and check if the class label is "license-plate"
        if confidence >= CONFIDENCE_THRESHOLD and class_label.lower() == "license-plate":
            # Append to the license plate list
            license_plate_list.append([xmin, ymin, xmax, ymax])
            
            # Draw bounding box and label for license plate
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), COLOR, 2)
            text = "License Plate: {:.2f}%".format(confidence * 100)
            cv2.putText(image, text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 2)
            
            # Optionally display the cropped number plate region
            if display:
                number_plate = image[ymin:ymax, xmin:xmax]
                cv2.imshow(f"License plate {len(license_plate_list)}", number_plate)

        # Check if the class label is "vehicle"
        if confidence >= CONFIDENCE_THRESHOLD and class_label.lower() == "vehicle":
            # Append to the vehicle list
            vehicle_list.append([xmin, ymin, xmax, ymax])

            # Draw bounding box and label for vehicle
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)  # Blue color for vehicle
            text = "Vehicle: {:.2f}%".format(confidence * 100)
            cv2.putText(image, text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    end = time.time()
    # Show time taken to detect number plates and vehicles
    print(f"Time to detect objects: {(end - start) * 1000:.0f} milliseconds")
    print(f"{len(license_plate_list)} License plate(s) have been detected.")
    print(f"{len(vehicle_list)} Vehicle(s) have been detected.")
    
    # Return both lists: all objects detected, license plates, and vehicles specifically
    return all_objects_list, license_plate_list, vehicle_list


def write_to_csv(file_path, number_plate_list):
    """
    Write the recognized license plates and their bounding boxes to a CSV file.
    """
    # Open the CSV file in append mode
    with open(CSV_FILE_PATH, mode='a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        
        # Write the header if it's a new file (optional)
        if os.stat(CSV_FILE_PATH).st_size == 0:
            csv_writer.writerow(["image_path", "box", "cleaned_text"])

        # Write each recognized license plate to the CSV file
        for plate_info in number_plate_list:
            # plate_info contains [box, cleaned_text]
            box = plate_info[0]
            cleaned_text = plate_info[1]
            
            # Write the file path, bounding box, and the recognized text to CSV
            csv_writer.writerow([file_path, box, cleaned_text])

def recognize_number_plates(image_or_path, reader, number_plate_list):
    """
    Recognize the text on the number plates and return the list with detected text.
    """
    start = time.time()

    # Read the image from path or use the provided image object
    image = cv2.imread(image_or_path) if isinstance(image_or_path, str) else image_or_path

    # Iterate over each detected number plate
    for i, box in enumerate(number_plate_list):
        # Assuming box is [xmin, ymin, xmax, ymax]
        xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]

        # Crop the number plate region from the image
        np_image = image[ymin:ymax, xmin:xmax]

        # Perform OCR on the cropped region
        detection = reader.readtext(np_image)
        text = detection[0][1] if detection else ""  # Get detected text or empty string if none
        cleaned_text = ''.join([char for char in text if char.isalnum()])  # Clean the text

        # Append the cleaned text to the bounding box
        number_plate_list[i] = [box, cleaned_text]

    end = time.time()
    # Show the time it took to recognize the number plates
    print(f"Time to recognize the number plates: {(end - start) * 1000:.0f} milliseconds")

    return number_plate_list
def find_license_plate_in_vehicle_bounds(license_plate_list, vehicle_list):
    """
    For each license plate, check if it falls within the bounds of a vehicle box.
    Separates license plates based on their associated vehicles.
    
    Returns:
    - A dictionary where each key is a vehicle bounding box and the value is a list of license plates within that vehicle's bounds.
    """
    plates_in_vehicles = {}

    # Loop over all vehicles
    for vehicle in vehicle_list:
        vehicle_xmin, vehicle_ymin, vehicle_xmax, vehicle_ymax = vehicle

        # Find license plates within this vehicle's bounds
        plates_in_vehicle = []
        for plate in license_plate_list:
            plate_xmin, plate_ymin, plate_xmax, plate_ymax = plate

            # Check if the license plate is within the vehicle bounds
            if (plate_xmin >= vehicle_xmin and plate_xmax <= vehicle_xmax and
                plate_ymin >= vehicle_ymin and plate_ymax <= vehicle_ymax):
                plates_in_vehicle.append(plate)

        # Add plates detected in this vehicle's bounds to the dictionary
        if plates_in_vehicle:
            plates_in_vehicles[(vehicle_xmin, vehicle_ymin, vehicle_xmax, vehicle_ymax)] = plates_in_vehicle

    return plates_in_vehicles


def filter_and_clean_text(text):
    """
    Clean and filter the OCR detected text.
    - Convert visually similar lowercase letters (c, o, s, v, w, x, z) to uppercase.
    - Remove any lowercase letters that don't have similar uppercase forms.
    - Retain only alphanumeric characters.
    
    Args:
    - text: The detected text string from OCR.
    
    Returns:
    - Cleaned text with valid uppercase letters and digits.
    """
    similar_letters = {'c', 'o', 's', 'v', 'w', 'x', 'z'}
    cleaned_text = []

    for char in text:
        if char.isdigit():  # Keep digits
            cleaned_text.append(char)
        elif char.isalpha():
            if char.isupper():  # Keep uppercase letters
                cleaned_text.append(char)
            elif char in similar_letters:  # Convert similar lowercase letters to uppercase
                cleaned_text.append(char.upper())
            # If it's a non-similar lowercase letter, it will be ignored (filtered out)

    return ''.join(cleaned_text)

def collect_character_confidences(ocr_results):
    """
    Collects all recognized characters and their confidence scores for each position.
    
    Args:
    - ocr_results: List of OCR results from multiple frames, each containing text and confidence.
    
    Returns:
    - A dictionary where each key is the position (0-5 for 6-character plates), and the value
      is another dictionary mapping characters to their confidence scores.
    """
    position_confidences = defaultdict(lambda: defaultdict(list))

    # Process each OCR result
    for result in ocr_results:
        text = result[1]  # Detected text
        confidence = result[2]  # Confidence score
        
        # Clean the text
        cleaned_text = filter_and_clean_text(text)
        
        # Only consider texts with at least 6 characters
        if len(cleaned_text) >= 6:
            # Track the first 6 characters of the cleaned text
            for i in range(6):
                char = cleaned_text[i]
                position_confidences[i][char].append(confidence)

    return position_confidences

def construct_most_likely_plate(position_confidences):
    """
    Constructs the most likely license plate by selecting the highest confidence characters
    for each of the 6 positions.
    
    Args:
    - position_confidences: Dictionary of characters and their confidences for each position.
    
    Returns:
    - The most likely 6-character license plate string.
    """
    most_likely_plate = ""

    for pos in range(6):
        if pos in position_confidences:
            # Select the character with the highest average confidence for this position
            most_likely_char = max(position_confidences[pos], key=lambda x: sum(position_confidences[pos][x]) / len(position_confidences[pos][x]))
            most_likely_plate += most_likely_char

    return most_likely_plate

def most_likely_license_plate(ocr_results):
    """
    Uses all recognized text to come up with the best 6-character combination of letters.
    
    Args:
    - ocr_results: List of OCR results from multiple frames.
    
    Returns:
    - The best 6-character combination based on the recognized text and confidence scores.
    """
    # Collect all character confidences by position (0-5 for 6-character plates)
    position_confidences = collect_character_confidences(ocr_results)

    # Construct the most likely license plate
    most_likely_plate = construct_most_likely_plate(position_confidences)

    return most_likely_plate



def process_license_plates_within_vehicles(image, license_plate_list, vehicle_list, reader):
    """
    Process the license plates and associate them with vehicles.
    For each license plate within a vehicle, perform OCR and apply a heuristic to determine the most likely license plate.
    
    Returns:
    - A dictionary where each vehicle is associated with the most likely license plate.
    """
    plates_in_vehicles = find_license_plate_in_vehicle_bounds(license_plate_list, vehicle_list)
    vehicle_plate_map = {}

    # Process each vehicle's license plates
    for vehicle, plates in plates_in_vehicles.items():
        ocr_results = []

        # Perform OCR on each plate associated with the vehicle
        for plate in plates:
            xmin, ymin, xmax, ymax = plate
            np_image = image[ymin:ymax, xmin:xmax]

            # Perform OCR on the cropped plate image
            detection = reader.readtext(np_image)
            if detection:
                ocr_results.extend(detection)

        # Use the heuristic to find the most likely license plate
        most_likely_plate = most_likely_license_plate(ocr_results)

        # Map the most likely license plate to the vehicle bounding box
        vehicle_plate_map[vehicle] = most_likely_plate

    return vehicle_plate_map
def calculate_iou(box1, box2):
    # Calculate the (x, y) coordinates of the intersection rectangle
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    # Compute the area of intersection
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Compute the area of both the boxes
    box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Compute the intersection over union (IoU)
    iou = interArea / float(box1Area + box2Area - interArea)

    return iou

# Class to track and stabilize license plate display
class LicensePlateTracker:
    def __init__(self, iou_threshold=0.3):
        self.tracked_plates = []  # List of currently tracked plates
        self.iou_threshold = iou_threshold

    def update(self, new_plates, frame, reader):
        """
        Update the tracked plates with new detections.
        Args:
            new_plates: List of new license plate bounding boxes detected in the current frame.
            frame: The current frame.
            reader: The EasyOCR reader.
        """
        new_tracked_plates = []
        for new_plate in new_plates:
            matched = False
            for tracked_plate in self.tracked_plates:
                iou = calculate_iou(new_plate, tracked_plate['box'])
                if iou > self.iou_threshold:
                    # Update the tracked plate's bounding box
                    tracked_plate['box'] = new_plate
                    # Keep adding new OCR results for this tracked plate
                    xmin, ymin, xmax, ymax = new_plate
                    cropped_plate = frame[ymin:ymax, xmin:xmax]
                    ocr_result = reader.readtext(cropped_plate)
                    if ocr_result:
                        tracked_plate['ocr_results'].extend(ocr_result)
                    new_tracked_plates.append(tracked_plate)
                    matched = True
                    break

            if not matched:
                # Perform OCR on the new plate
                xmin, ymin, xmax, ymax = new_plate
                cropped_plate = frame[ymin:ymax, xmin:xmax]
                ocr_result = reader.readtext(cropped_plate)
                if ocr_result:
                    new_tracked_plates.append({
                        'box': new_plate,
                        'ocr_results': ocr_result,  # Collect the OCR results for this plate
                        'frame_count': 0  # Keep track of how long we've seen this plate
                    })

        self.tracked_plates = new_tracked_plates

    def draw_plates(self, frame):
        """
        Draw the tracked plates on the frame.
        """
        for plate in self.tracked_plates:
            box = plate['box']
            ocr_results = plate['ocr_results']
            
            # Use the updated heuristic to find the most likely plate
            most_likely_plate = most_likely_license_plate(ocr_results)

            # Draw the bounding box
            xmin, ymin, xmax, ymax = box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            # Draw the license plate text in hot pink (RGB: 255, 105, 180) underneath the bounding box
            hot_pink = (255, 105, 180)
            text_position_y = ymax + 20  # Set the text position just below the bounding box
            cv2.putText(frame, most_likely_plate, (xmin, text_position_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, hot_pink, 2)


def process_image(file_path, model, reader):
    """
    Process an image file for detecting and recognizing number plates.
    """
    # Read the image
    image = cv2.imread(file_path)
    print("Processing the image...")

    # Detect number plates in the image
    _, number_plate_list = detect_number_plates(image, model, display=True)
    
    if not number_plate_list or len(number_plate_list) == 0:
        print("No number plates were detected.")
    else:
        print(f"Detected number plates: {number_plate_list}")

        # Run OCR on the detected number plates
        if number_plate_list:
            print("Starting OCR recognition...")
            number_plate_list = recognize_number_plates(file_path, reader, number_plate_list)

            # Write the recognized license plates to the CSV file
            write_to_csv(file_path, number_plate_list)

            # Display OCR results on the image
            for entry in number_plate_list:
                box = entry[0]
                text = entry[1]

                # Extract xmin, ymin, xmax, ymax from the box
                xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]

                # Draw text above the detected bounding box
                cv2.putText(image, text, (xmin, ymin - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Show the final image with OCR results
            cv2.imshow('Image', image)
            cv2.waitKey(0)

def process_video(file_path, model, reader):
    """
    Process a video file for detecting and recognizing number plates.
    Constantly displays the most likely license plate number on the screen.
    """
    print("Processing the video...")
    video_cap = cv2.VideoCapture(file_path)

    # Grab the width and height of the video stream
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))

    # Initialize the FourCC and a video writer object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter("output.mp4", fourcc, fps, (frame_width, frame_height))

    # Initialize the license plate tracker
    plate_tracker = LicensePlateTracker(iou_threshold=0.3)

    while True:
        # Start time to compute the fps
        start = time.time()

        success, frame = video_cap.read()

        if not success:
            print("There are no more frames to process. Exiting the script...")
            break

        # Detect number plates and vehicles in the frame
        _, number_plate_list, _ = detect_number_plates(frame, model)

        if number_plate_list:
            # Update the tracker with new plates
            plate_tracker.update(number_plate_list, frame, reader)

        # Draw the tracked plates and their most likely license plate text
        plate_tracker.draw_plates(frame)

        # End time to compute the fps
        end = time.time()

        # Calculate fps and draw it on the frame
        fps_text = f"FPS: {1 / (end - start):.2f}"
        cv2.putText(frame, fps_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)

        # Display the frame with vehicle and license plate annotations
        cv2.imshow("Output", frame)

        # Write the frame to the output file
        writer.write(frame)

        # Exit the loop if 'q' key is pressed
        if cv2.waitKey(10) == ord("q"):
            break

    # Release the video capture and writer
    video_cap.release()
    writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Load the YOLO model
    model = YOLO("train/license_plate_detection13/weights/best.pt")

    # Initialize the EasyOCR reader
    try:
        reader = Reader(['en'], gpu=True)
        print("OCR reader initialized successfully.")
    except Exception as e:
        print(f"Error initializing OCR reader: {str(e)}")

    # Path to an image or a video file
    file_path = "IMG_2180.mov"

    # Extract the file extension
    _, file_extension = os.path.splitext(file_path)

    # Process based on file type
    if file_extension in ['.jpg', '.jpeg', '.png']:
        process_image(file_path, model, reader)
    elif file_extension in ['.mp4', '.mkv', '.avi', '.wmv', '.mov']:
        process_video(file_path, model, reader)
    else:
        print("Unsupported file format.")
