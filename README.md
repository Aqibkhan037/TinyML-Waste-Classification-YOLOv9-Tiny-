🗑️ Trash Classification with YOLOv9 Tiny

This project applies YOLOv9 Tiny for real-time waste classification using the TrashNet dataset. The goal is to build a lightweight, optimized model that can run efficiently on edge devices like Raspberry Pi and smartphones.
______________________________________________________________________________________________________________________________________________________________________________________________________________________________

🚀 Features

Object detection for glass, metal, plastic, paper, and other waste types.

Trained on TrashNet dataset, preprocessed via Roboflow.

Optimized for TinyML deployment through:

YOLOv9 Tiny architecture (fast + lightweight).

Pruning (35% size reduction, faster inference).

Quantization (75% smaller, INT8 deployment).

Achieves ~72% mAP with real-time inference (~7ms/frame) after optimization.
______________________________________________________________________________________________________________________________________________________________________________________________________________________________

🛠️ Tech Stack

Deep Learning Framework: PyTorch

Model: YOLOv9 Tiny (Ultralytics)

Dataset: TrashNet (via Roboflow)

Optimization: Pruning + Quantization

Deployment: Edge devices (Raspberry Pi, smartphones)
______________________________________________________________________________________________________________________________________________________________________________________________________________________________

📊 Training Workflow

Dataset Preprocessing (Roboflow)

Resized to 640x640

Labeled & split (70% train / 20% val / 10% test)

Exported in YOLO format

YOLOv9 Tiny Training

from ultralytics import YOLO

# Load YOLOv9 Tiny
model = YOLO('yolov9-tiny.yaml')  

# Train
results = model.train(data='dataset.yaml', epochs=100, imgsz=640, batch=16)  

# Export
model.export(format='torchscript', weights='yolov9_trashnet.pt')  


Optimization

Pruning → 35% model reduction, faster inference

Quantization → INT8 model, 75% smaller, ~7ms inference
______________________________________________________________________________________________________________________________________________________________________________________________________________________________

📈 Results

Accuracy (mAP): ~72% (69% post-pruning, ~71% post-quantization)

Speed: From 20ms → 7ms per frame

Model Size: Reduced by 75% with quantization
______________________________________________________________________________________________________________________________________________________________________________________________________________________________

🌍 Applications

Smart waste bins for automatic sorting

Mobile apps for recycling awareness

IoT / Edge AI for real-time environmental monitoring

______________________________________________________________________________________________________________________________________________________________________________________________________________________________

📚 References

Han, S. et al., Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding

TensorFlow, Quantization: Speed Up and Compress Your Models

YOLOv9 Ultralytics
______________________________________________________________________________________________________________________________________________________________________________________________________________________________
👨‍💻 Author

Developed as part of research into TinyML for environmental sustainability 🌱
aqibkhan1528000@gmail.com
