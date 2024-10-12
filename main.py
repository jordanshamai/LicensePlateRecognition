# import cv2
# import os
# from ultralytics import YOLO
# from detect import detect_number_plates
# from ocr import initialize_ocr
# from tracker import LicensePlateTracker

# def process_image(file_path, model, reader):
#     print(f"Processing image: {file_path}")
    
#     # Read the image
#     image = cv2.imread(file_path)
#     if image is None:
#         print(f"Error: Unable to load image at {file_path}")
#         return

#     print("Detecting number plates...")
    
#     # Initialize tracker
#     plate_tracker = LicensePlateTracker(iou_threshold=0.3)
#     run_yolo = True

#     while run_yolo:
#         # Detect number plates using YOLO
#         _, number_plate_list, _ = detect_number_plates(image, model, display=True, tracker=plate_tracker)
#         print(f"Detected number plates: {number_plate_list}")

#         if number_plate_list:
#             # Update the tracker with detected plates
#             plate_tracker.update(number_plate_list, image, reader)

#             # Draw plates onto the image
#             plate_tracker.draw_plates(image)

#             # Stop running YOLO once plates are detected and processed
#             run_yolo = False
#         else:
#             print("No new plates detected, continuing YOLO...")

#     # Display the final image with results
#     cv2.imshow('Image', image)
#     cv2.waitKey(0)

# def process_video(file_path, model, reader):
#     print(f"Processing video: {file_path}")
    
#     video_cap = cv2.VideoCapture(file_path)
#     frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = int(video_cap.get(cv2.CAP_PROP_FPS))
#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     writer = cv2.VideoWriter("output.mp4", fourcc, fps, (frame_width, frame_height))
    
#     plate_tracker = LicensePlateTracker(iou_threshold=0.3, max_disappeared=20)  # Adjust max_disappeared threshold

#     while True:
#         success, frame = video_cap.read()
#         if not success:
#             break

#         print("Detecting number plates in video frame...")

#         # Detect number plates using YOLO if no tracked plates or re-run after threshold
#         if plate_tracker.should_run_yolo():
#             _, number_plate_list, _ = detect_number_plates(frame, model, display=False, tracker=plate_tracker)
#             print(f"Detected number plates: {number_plate_list}")
#             if len(number_plate_list) == 0:
#                 plate_tracker.increment_disappeared_count()  # Increase counter for no detection
#         else:
#             print("Using tracked plates, skipping YOLO detection.")

#         # Update tracker with detected plates or process if none
#         if number_plate_list:
#             plate_tracker.update(number_plate_list, frame, reader)
#         else:
#             plate_tracker.handle_disappeared_plates()  # Handle plates that disappear

#         # Draw plates on the current frame and remove bounding boxes if plates disappear
#         plate_tracker.draw_plates(frame)

#         # Write the frame to the output video
#         writer.write(frame)
#         cv2.imshow("Output", frame)

#         # Press 'q' to quit the video processing
#         if cv2.waitKey(10) == ord("q"):
#             break

#     video_cap.release()
#     writer.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     model = YOLO("best.pt")
    
#     # Initialize PaddleOCR in ocr.py
#     reader = initialize_ocr()
    
#     file_path = "Videos/Video.mov"
#     _, file_extension = os.path.splitext(file_path)

#     if file_extension in ['.jpg', '.jpeg', '.png']:
#         process_image(file_path, model, reader)
#     elif file_extension in ['.mp4', '.mkv', '.avi', '.wmv', '.mov']:
#         process_video(file_path, model, reader)
#     else:
#         print("Unsupported file format.")


import cv2
import os
from ultralytics import YOLO
from detect import detect_number_plates
from ocr import initialize_ocr  # Only need initialize_ocr, the tracker will handle the rest
from tracker import LicensePlateTracker

def process_image(file_path, model, reader):
    print(f"Processing image: {file_path}")
    
    # Read the image
    image = cv2.imread(file_path)
    if image is None:
        print(f"Error: Unable to load image at {file_path}")
        return

    print("Detecting number plates...")
    # Detect number plates using YOLO
    _, number_plate_list, _ = detect_number_plates(image, model, display=True)
    print(f"Detected number plates: {number_plate_list}")

    if number_plate_list:
        # Initialize tracker
        plate_tracker = LicensePlateTracker(iou_threshold=0.3)
        
        # Update the tracker with detected plates
        plate_tracker.update(number_plate_list, image, reader)

        # Draw plates onto the image
        plate_tracker.draw_plates(image)

        # Display the final image with results
        cv2.imshow('Image', image)
        cv2.waitKey(0)
    else:
        print("No number plates detected.")

def process_video(file_path, model, reader):
    print(f"Processing video: {file_path}")
    
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

        print("Detecting number plates in video frame...")
        _, number_plate_list, _ = detect_number_plates(frame, model)
        print(f"Detected number plates: {number_plate_list}")

        if number_plate_list:
            # Update the tracker with detected plates in the current frame
            plate_tracker.update(number_plate_list, frame, reader)

        # Draw plates on the current frame
        plate_tracker.draw_plates(frame)

        # Write the frame to the output video
        writer.write(frame)
        cv2.imshow("Output", frame)

        # Press 'q' to quit the video processing
        if cv2.waitKey(10) == ord("q"):
            break

    video_cap.release()
    writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model = YOLO("best.pt")
    
    # Initialize PaddleOCR in ocr.py
    reader = initialize_ocr()
    
    file_path = "Videos/Video.mov"
    _, file_extension = os.path.splitext(file_path)

    if file_extension in ['.jpg', '.jpeg', '.png']:
        process_image(file_path, model, reader)
    elif file_extension in ['.mp4', '.mkv', '.avi', '.wmv', '.mov']:
        process_video(file_path, model, reader)
    else:
        print("Unsupported file format.")
#Implementing better detection logic:

# import cv2
# import os
# from ultralytics import YOLO
# from detect import detect_number_plates
# from ocr import initialize_ocr
# from tracker import LicensePlateTracker

# # Set confidence thresholds for both YOLO and OCR
# YOLO_CONFIDENCE_THRESHOLD = 0.6
# OCR_CONFIDENCE_THRESHOLD = 0.8

# def process_image(file_path, model, reader):
#     print(f"Processing image: {file_path}")
    
#     # Read the image
#     image = cv2.imread(file_path)
#     if image is None:
#         print(f"Error: Unable to load image at {file_path}")
#         return

#     print("Detecting number plates...")
#     while True:
#         # Detect number plates using YOLO
#         _, number_plate_list, _ = detect_number_plates(image, model, display=True)
#         print(f"Detected number plates: {number_plate_list}")

#         if not number_plate_list:
#             print("No number plates detected.")
#             break  # If no plates, break the loop

#         for plate in number_plate_list:
#             xmin, ymin, xmax, ymax = plate  # Get coordinates of detected plate
#             cropped_plate = image[ymin:ymax, xmin:xmax]

#             print("Running OCR...")
#             # Perform OCR on the detected plate
#             detection = reader.readtext(cropped_plate)
#             if detection:
#                 text, confidence = detection[0][1], detection[0][2]  # Extract text and OCR confidence
#                 print(f"OCR Text: {text}, Confidence: {confidence}")

#                 if confidence >= OCR_CONFIDENCE_THRESHOLD:
#                     print(f"High OCR confidence ({confidence}), stopping detection.")
#                     # Draw the OCR result on the image
#                     cv2.putText(image, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#                     # Display the final image with results
#                     cv2.imshow('Image', image)
#                     cv2.waitKey(0)
#                     return  # Stop further processing since we found a good plate
#                 else:
#                     print(f"Low OCR confidence ({confidence}), running detection again...")
#                     # If OCR confidence is low, continue running YOLO to find other plates
#             else:
#                 print("No text detected in OCR, running detection again...")

# def process_video(file_path, model, reader):
#     print(f"Processing video: {file_path}")
    
#     video_cap = cv2.VideoCapture(file_path)
#     frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = int(video_cap.get(cv2.CAP_PROP_FPS))
#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     writer = cv2.VideoWriter("output.mp4", fourcc, fps, (frame_width, frame_height))
#     plate_tracker = LicensePlateTracker(iou_threshold=0.3)

#     while True:
#         success, frame = video_cap.read()
#         if not success:
#             break

#         print("Detecting number plates in video frame...")
#         _, number_plate_list, _ = detect_number_plates(frame, model, reader)
#         print(f"Detected number plates: {number_plate_list}")

#         if number_plate_list:
#             for plate in number_plate_list:
#                 xmin, ymin, xmax, ymax = plate  # Get coordinates of detected plate
#                 cropped_plate = frame[ymin:ymax, xmin:xmax]

#                 print("Running OCR...")
#                 # Perform OCR on the detected plate
#                 detection = reader.readtext(cropped_plate)
#                 if detection:
#                     text, confidence = detection[0][1], detection[0][2]  # Extract text and OCR confidence
#                     print(f"OCR Text: {text}, Confidence: {confidence}")

#                     if confidence >= OCR_CONFIDENCE_THRESHOLD:
#                         print(f"High OCR confidence ({confidence}), updating tracker.")
#                         # Update the tracker with the high-confidence plate
#                         plate_tracker.update(number_plate_list, frame, reader)

#                     # Draw plates on the current frame
#                     plate_tracker.draw_plates(frame)

#                 # Write the frame to the output video
#                 writer.write(frame)
#                 cv2.imshow("Output", frame)

#         # Press 'q' to quit the video processing
#         if cv2.waitKey(10) == ord("q"):
#             break

#     video_cap.release()
#     writer.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     model = YOLO("best.pt")
    
#     # Initialize PaddleOCR in ocr.py
#     reader = initialize_ocr()
    
#     file_path = "Videos/Video.mov"
#     _, file_extension = os.path.splitext(file_path)

#     if file_extension in ['.jpg', '.jpeg', '.png']:
#         process_image(file_path, model, reader)
#     elif file_extension in ['.mp4', '.mkv', '.avi', '.wmv', '.mov']:
#         process_video(file_path, model, reader)
#     else:
#         print("Unsupported file format.")
