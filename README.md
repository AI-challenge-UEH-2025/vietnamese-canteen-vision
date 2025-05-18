# Vietnamese Canteen Vision: AI-Powered Food Recognition System
<p align="center">
  <a href="https://youtu.be/w-ruyauu5rc?si=OVgKPpun4WrvD4pU" target="_blank">
    <img src="https://raw.githubusercontent.com/AI-challenge-UEH-2025/vietnamese-canteen-vision/main/web_ui/static/img/banner.jpg" alt="Vietnamese Canteen Vision Banner" width="800"/>
  </a>
    <br/> <br/> <a href="https://www.python.org/downloads/" target="_blank"><img alt="Python 3.8+" src="https://img.shields.io/badge/python-3.8+-blue.svg"/></a>
  <a href="https://www.tensorflow.org/" target="_blank"><img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-2.9+-orange.svg"/></a>
  <a href="https://github.com/ultralytics/ultralytics" target="_blank"><img alt="YOLOv8" src="https://img.shields.io/badge/YOLO-v8-darkgreen.svg"/></a>
  <a href="https://opensource.org/licenses/MIT" target="_blank"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"/></a>
</p>

## 📋 Overview

CanteenVision is an intelligent system that automates billing for canteen food trays. Using computer vision and deep learning, the system detects and classifies Vietnamese food items on a tray, calculates the total cost, and provides nutritional information - all in real-time.

Built for the AI Challenge 3ITECH 2025, this project demonstrates how AI can streamline canteen operations while enhancing the student experience.

## ✨ Key Features

- **🔍 Advanced Food Detection**: Uses YOLOv8 to locate and crop individual food items on a tray
- **🍲 Vietnamese Food Classification**: Employs a fine-tuned ResNet50V2 model to identify 41 different Vietnamese food items
- **💰 Automated Billing**: Calculates the total cost based on detected items
- **🥗 Nutritional Analysis**: Provides calorie content and meal balance feedback
- **🖥️ Responsive Web Interface**: User-friendly design that works on both desktop and mobile devices
- **✏️ Manual Correction**: Allows for easy adjustment of misidentified items
- **📊 Transaction History**: Keeps records of past purchases
- **🌙 Dark/Light Mode**: Interface adapts to user preference

## 🖼️ Screenshots

<div align="center">
  <img src="https://raw.githubusercontent.com/AI-challenge-UEH-2025/vietnamese-canteen-vision/main/web_ui/static/img/calo-info.jpg" width="400px">
  <img src="https://raw.githubusercontent.com/AI-challenge-UEH-2025/vietnamese-canteen-vision/main/web_ui/static/img/manually-added.jpg" width="400px">
  <img src="https://raw.githubusercontent.com/AI-challenge-UEH-2025/vietnamese-canteen-vision/main/web_ui/static/img/history.jpg" width="400px">
  <img src="https://raw.githubusercontent.com/AI-challenge-UEH-2025/vietnamese-canteen-vision/main/web_ui/static/img/menu-info.jpg" width="400px">
  <img src="https://raw.githubusercontent.com/AI-challenge-UEH-2025/vietnamese-canteen-vision/main/web_ui/static/img/setting.jpg" width="400px">
  <img src="https://raw.githubusercontent.com/AI-challenge-UEH-2025/vietnamese-canteen-vision/main/web_ui/static/img/dark-mode.jpg" width="400px">
</div>

## 🛠️ Technology Stack

- **Backend**: Python, Flask
- **Computer Vision**: YOLOv8, OpenCV
- **Deep Learning**: TensorFlow/Keras with ResNet50V2
- **Frontend**: HTML, CSS, JavaScript
- **Data Storage**: CSV-based menu system

## 🏗️ System Architecture

The system follows a pipeline architecture:

1. **Detection**: YOLOv8 model detects food items and crops individual items
2. **Classification**: ResNet50V2 model classifies each food item
3. **Verification**: Cross-validates YOLO and ResNet results for improved accuracy
4. **Billing**: Calculates total cost based on identified items
5. **Presentation**: Displays results through an interactive web interface

## 🍽️ Supported Food Items

The system can recognize 41 different Vietnamese food items including:

- banh mi (Vietnamese sandwich)
- bap cai luoc (boiled cabbage)
- ca chien (fried fish)
- canh chua (sour soup)
- com (rice)
- ga chien (fried chicken)
- rau muong (morning glory vegetable)
- thit kho (braised pork)
- trung chien (fried egg)
- and many more...

## 🚀 Installation and Setup

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for inference)
- Web camera or image input source

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/SPARK-UNI/vietnamese-canteen-vision.git
   cd vietnamese-canteen-vision
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the web server:
   ```bash
   python web_server.py
   ```

4. Access the interface at http://localhost:5000

### Docker Installation (Alternative)

```bash
# Build the Docker image
docker build -t canteenvision .

# Run the container
docker run -p 5000:5000 canteenvision
```

## 💻 Usage

1. **Upload a Food Tray Image**:
   - Click "Select Image" or drag and drop an image of a food tray
   - Alternatively, use the "Use Camera" button to capture a live image

2. **Analyze the Image**:
   - Click "Analyze Food" to process the image
   - The system will detect and classify each food item

3. **Review Results**:
   - View detected items, prices, and calorie information
   - Make corrections to misidentified items if needed
   - Check the nutritional balance and suggestions

4. **Complete the Order**:
   - Click "Complete Order" to finalize and save the transaction

## 🧠 AI Model Details

### YOLOv8 Object Detection

The object detection model was trained using YOLOv8 on a custom dataset of Vietnamese food trays:

- **Architecture**: YOLOv8-seg (with segmentation capabilities)
- **Training Data**: 1,000+ images of food trays with bounding box annotations
- **Performance**: 92% mAP@0.5 on validation set
- **Inference Time**: ~150ms per image on GPU

### ResNet50V2 Classification

The classification model was fine-tuned using transfer learning:

- **Base Architecture**: ResNet50V2 pre-trained on ImageNet
- **Fine-tuning**: Last 30 layers were unfrozen and fine-tuned
- **Training Data**: 4,000+ images across 41 food classes
- **Augmentation**: Rotation, flipping, brightness adjustment, etc.
- **Performance**: 89% accuracy on validation set

## 📊 Project Structure

```
vietnamese-canteen-vision/
│
├── data/                        # Dataset directory
│
├── models/                      # AI model files
│   ├── cnn/                     # CNN model directory
│   ├── best.pt                  # YOLO best model
│   ├── cnn.h5                   # CNN model
│   ├── resnet50v2_finetuned.h5  # Fine-tuned ResNet model
│   └── resnet50v2_initial.h5    # Initial ResNet model
│
├── src/                         # Python source code
│   ├── __pycache__/             # Python cache
│   ├── augmentation.py          # Data augmentation utilities
│   ├── billing.py               # Billing logic
│   ├── classify.py              # Food classification
│   ├── detect.py                # Food detection
│   ├── gui.py                   # GUI application
│   ├── train_resnet.py          # ResNet training script
│   └── train_yolo.py            # YOLO training script
│
├── web_ui/                      # Web interface
│   ├── static/                  # CSS, JS, images
│   ├── templates/               # HTML templates
│   └── .gitignore               # Git ignore file
│
├── app.py                       # Flask application
├── main.py                      # Main entry point
├── requirements.txt             # Dependencies
├── web_server.py                # Server startup script
└── README.md                    # This file
```

## 📈 Future Improvements

- [ ] Integration with payment systems
- [ ] Mobile app development
- [ ] Support for additional languages
- [ ] Expanded food database
- [ ] Integration with inventory management
- [ ] Dietary restriction warnings
- [ ] Personalized recommendations based on previous orders

## 👥 Contributors

- [AI Challenge UEH 2025 Team](https://github.com/AI-challenge-UEH-2025) - Project Team

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- University of Economics Ho Chi Minh City (UEH) for supporting this project
- Faculty supervisors at UEH for project guidance
- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLOv8 implementation
- [TensorFlow](https://www.tensorflow.org/) team for the deep learning framework
- All students and staff who contributed to data collection and testing

---

<div align="center">
  <p>Made with ❤️ for the AI Challenge 3ITECH 2025</p>
  <p>
    <a href="https://github.com/AI-challenge-UEH-2025/vietnamese-canteen-vision/issues">Report Bug</a> ·
    <a href="https://github.com/AI-challenge-UEH-2025/vietnamese-canteen-vision/issues">Request Feature</a>
  </p>
</div>
