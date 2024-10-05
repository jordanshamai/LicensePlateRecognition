from ultralytics import YOLO

def train_model():
    # Define the path to your dataset YAML file and the model to use
    DATA_PATH = r'C:\Users\jorda\OneDrive\computer science\MachineLearning\TrainModel\data.yaml'
    MODEL_PATH = 'yolov8n.pt'  # Path to the YOLOv8 model weights

    # Load the YOLOv8 model
    model = YOLO(MODEL_PATH)

    # Train the model using GPU (CUDA)
    model.train(
        data=DATA_PATH,          # Path to the data configuration file
        epochs=100,              # Number of training epochs
        imgsz=640,               # Image size (adjust based on your hardware)
        batch=16,                # Batch size (adjust based on your GPU's memory)
        device=0,                # Use GPU with device index 0 (use '0' for the first GPU, '1' for the second, etc.)
        name='license_plate_detection',  # Name of the training run
        project='runs/train'     # Directory to save training results
    )

    # Evaluate the model on the validation set using GPU
    metrics = model.val(device=0)

    # Print the evaluation metrics
    print(metrics)

    # Run predictions on the test data using GPU
    test_images_path = 'test/images'  # Adjust this path to your test images directory
    results = model.predict(source=test_images_path, save=True, conf=0.25, device=0)

    # Print summary of the results
    print(f"Predictions on test data completed. Results saved in {results.save_dir}")

    # Save the best model
    model.export(format='onnx')  # Export the model to ONNX format or any other format you prefer

if __name__ == '__main__':
    train_model()
