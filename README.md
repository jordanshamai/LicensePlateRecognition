# License Plate Detection & OCR System  
**Built with YOLOv8 + PaddleOCR (formerly EasyOCR) — Fine-Tuned with Roboflow**

> An end-to-end system for detecting and recognizing license plates in real-world conditions using state-of-the-art computer vision and OCR.

---

## What It Does

This project combines **YOLOv8** for robust license plate detection with **PaddleOCR** for high-accuracy recognition of plate numbers. Originally prototyped with **EasyOCR**, we transitioned to **PaddleOCR** for better fine-tuning and real-world performance.

---

## Features

- **YOLOv8** custom-trained for license plate detection  
- **PaddleOCR** used for text recognition, fine-tuned with private data  
- **EasyOCR** was initially tested for OCR performance  
- Designed for modular use in apps, edge devices, or surveillance systems  
- Built with scalability and deployment in mind

---

## Model Training

### Object Detection
- Framework: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- Dataset: Fine-tuned with annotated license plate datasets from [Roboflow Universe](https://universe.roboflow.com/search?q=class%3A%22license+plate%22)

### OCR
- Initial prototype: [EasyOCR](https://github.com/JaidedAI/EasyOCR)  
- Final system: [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) with private fine-tuned dataset

---

## How It Works
[ Image / Video Frame ] 
            ↓ 
[ YOLOv8 Detector ]
            ↓      
[ Cropped License Plate Region ] 
            ↓ 
[ PaddleOCR Text Recognition ] 
            ↓ 
[ Plate Number Output ]


---

## Stack

| Component     | Tech                      |
|---------------|---------------------------|
| Detection     | YOLOv8 (Ultralytics)      |
| OCR           | PaddleOCR (final), EasyOCR (initial) |
| Dataset Mgmt  | Roboflow + private sets   |
| Language      | Python                    |
| Inference     | OpenCV, Torch, ONNX (optional) |

---

## Quick Start

```bash
git clone https://github.com/yourname/license-plate-ocr
cd license-plate-ocr

# Install dependencies
pip install -r requirements.txt

# Run detection + OCR on an image
python detect_and_ocr.py --source sample.jpg
```
## Acknowledgments

- Thanks to [Roboflow](https://roboflow.com/) for dataset hosting and augmentation tools  
- Built using [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) and [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)

---

## Future Work

- Real-time multi-frame tracking & plate verification
- Model quantization and fine tuning.
- REST API for license plate submission and matching  
- Integration with edge devices for smart surveillance and IoT systems  

---

## Contact

**Author:** Jordan Shamai  
**Email:** [jordan.shamai04@gmail.com](mailto:jordan.shamai04@gmail.com)  
**Project Type:** Research Prototype  

