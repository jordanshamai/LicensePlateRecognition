from ultralytics import YOLO

def test_model():
    # Path to the trained YOLOv8 model
    MODEL_PATH = 'runs/train/license_plate_detection/weights/yolov8n.pt'  # Update this to the correct path of your best model

    # Load the trained YOLOv8 model
    model = YOLO(MODEL_PATH)

    # Define the path to your test images directory
    TEST_IMAGES_PATH = 'test/images'  # Update this path to point to your test images directory

    # Run predictions on the test data using GPU (set device=0 to use the first GPU)
    results = model.predict(
        source=TEST_IMAGES_PATH,  # Path to test images
        save=True,                # Save the results (images with bounding boxes)
        conf=0.25,                # Confidence threshold for predictions
        device=0                  # Use GPU (change to 'cpu' if no GPU is available)
    )

    # Print summary of the results
    print(f"Predictions on test data completed. Results saved in {results.save_dir}")

if __name__ == '__main__':
    test_model()
