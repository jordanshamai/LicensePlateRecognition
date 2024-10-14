import cv2
import os
from ultralytics import YOLO
from detect import detect_number_plates
from ocr import initialize_ocr
from tracker import LicensePlateTracker

def process_image(file_path, model, reader):
    print(f"Processing image: {file_path}")
    
    # Read the image
    image = cv2.imread(file_path)
    if image is None:
        print(f"Error: Unable to load image at {file_path}")
        return

    print("Detecting number plates...")
    
    # Initialize tracker
    plate_tracker = LicensePlateTracker(iou_threshold=0.3)
    run_yolo = True

    while run_yolo:
        # Detect number plates using YOLO
        _, number_plate_list, _ = detect_number_plates(image, model, display=True, tracker=plate_tracker)
        print(f"Detected number plates: {number_plate_list}")

        if number_plate_list:
            # Update the tracker with detected plates
            plate_tracker.update(number_plate_list, image, reader)

            # Draw plates onto the image
            plate_tracker.draw_plates(image)

            # Stop running YOLO once plates are detected and processed
            run_yolo = False
        else:
            print("No new plates detected, continuing YOLO...")

    # Display the final image with results
    cv2.imshow('Image', image)
    cv2.waitKey(0)

def process_video(file_path, model, reader):
    print(f"Processing video: {file_path}")
    
    video_cap = cv2.VideoCapture(file_path)
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter("output.mp4", fourcc, fps, (frame_width, frame_height))
    
    plate_tracker = LicensePlateTracker(iou_threshold=0.3, max_disappeared=20)

    while True:
        success, frame = video_cap.read()
        if not success:
            break

        print("Detecting number plates in video frame...")

        # Detect number plates using YOLO
        if plate_tracker.should_run_yolo():
            _, number_plate_list, _ = detect_number_plates(frame, model, display=False)
            print(f"Detected number plates: {number_plate_list}")

            # If no plates detected, increment disappeared count
            if len(number_plate_list) == 0:
                plate_tracker.increment_disappeared_count()
        else:
            print("Using tracked plates, skipping YOLO detection.")

        # Process detected plates
        if number_plate_list:
            for plate in number_plate_list:
                # Extract the bounding box for the detected plate
                xmin, ymin, xmax, ymax = plate
                cropped_plate = frame[ymin:ymax, xmin:xmax]  # Save the plate image in memory

                # Run OCR on the cropped plate image
                print("Running OCR on detected license plate...")
                ocr_result = reader.ocr(cropped_plate)

                # Ensure that ocr_result is not None before processing
                print("Printing OCR Result:")
                print(ocr_result)
                if ocr_result== [None]:
                    print("No valid OCR result detected, skipping OCR for this plate.")
                else:
                    most_likely_plate = ocr_result[0][1][0]  # Extract the most likely text from OCR
                    print(f"Most likely plate: {most_likely_plate}")

                    # Display the result on the frame
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    cv2.putText(frame, most_likely_plate, (xmin, ymax + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 105, 180), 2)
                
                    

            # Update the tracker with detected plates
            plate_tracker.update(number_plate_list, frame, reader)
        else:
            plate_tracker.handle_disappeared_plates()

        # Draw the plates on the frame
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
        print("Image detected. Use process_image for still images.")
    elif file_extension in ['.mp4', '.mkv', '.avi', '.wmv', '.mov']:
        process_video(file_path, model, reader)
    else:
        print("Unsupported file format.")