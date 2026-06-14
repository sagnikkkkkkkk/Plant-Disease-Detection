<div align="center">

# 🌱 AgroVision AI
## Intelligent Plant Disease Detection & Diagnosis using Deep Learning

<img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python"/>
<img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=for-the-badge&logo=pytorch"/>
<img src="https://img.shields.io/badge/MobileNetV2-Transfer%20Learning-orange?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Accuracy-95%25-success?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Domain-Precision%20Agriculture-green?style=for-the-badge"/>

### 🚀 Transforming Agriculture with Artificial Intelligence

*An advanced Computer Vision system that detects plant diseases from leaf images with high accuracy using Transfer Learning and Deep Learning.*

</div>

---

# 📖 Overview

Agriculture feeds the world, yet plant diseases continue to cause billions of dollars in crop losses every year.

**AgroVision AI** is an intelligent plant disease diagnosis system designed to assist farmers, researchers, and agricultural experts in identifying diseases directly from leaf images using state-of-the-art Deep Learning techniques.

By leveraging **MobileNetV2 Transfer Learning**, the system automatically analyzes plant leaves, predicts diseases, estimates confidence levels, and provides treatment recommendations.

---

# 🎯 Project Highlights

✨ AI-Powered Disease Detection

✨ MobileNetV2 Transfer Learning

✨ Automated Image Classification

✨ Real-Time Prediction Pipeline

✨ Disease Information & Treatment Suggestions

✨ Top-5 Probability Analysis

✨ High Accuracy (~95%)

✨ Lightweight & Deployment Ready

✨ Agricultural Decision Support System

---

# 🌍 Why This Project Matters

Plant diseases often remain undetected until significant crop damage has occurred.

Traditional diagnosis:

❌ Time-consuming

❌ Expert-dependent

❌ Expensive

❌ Not scalable

AgroVision AI solves these problems through:

✅ Instant Disease Identification

✅ Automated Analysis

✅ Reduced Human Error

✅ Early Disease Intervention

✅ Improved Agricultural Productivity

---

# 🏗 System Architecture

```text
                    ┌───────────────────┐
                    │  Leaf Image Input │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Image Processing  │
                    │ Resize + Normalize│
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ MobileNetV2 CNN   │
                    │ Feature Extractor │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Classification    │
                    │ Layer             │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Disease Prediction│
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Confidence Score  │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Treatment Advice  │
                    └───────────────────┘
```

---

# 🧠 AI Model

## Deep Learning Backbone

**MobileNetV2**

Why MobileNetV2?

- Lightweight architecture
- Fast inference
- Low memory footprint
- Excellent transfer learning performance
- Suitable for mobile and edge deployment

### Transfer Learning Strategy

Instead of training from scratch:

- Pretrained ImageNet weights utilized
- Feature extraction layers frozen
- Custom classification head added
- Fine-tuned on PlantVillage dataset

This significantly reduces training time while improving accuracy.

---

# 📊 Dataset Information

## PlantVillage Dataset

The model is trained on a multi-class agricultural dataset containing thousands of labeled leaf images.

### Supported Classes

#### 🍅 Tomato

- Bacterial Spot
- Early Blight
- Late Blight
- Leaf Mold
- Septoria Leaf Spot
- Spider Mites
- Target Spot
- Yellow Leaf Curl Virus
- Mosaic Virus
- Healthy

#### 🥔 Potato

- Early Blight
- Late Blight
- Healthy

#### 🫑 Bell Pepper

- Bacterial Spot
- Healthy

---

# ⚙️ Tech Stack

| Category | Technology |
|-----------|------------|
| Language | Python |
| Deep Learning | PyTorch |
| CNN Model | MobileNetV2 |
| Data Processing | NumPy |
| Visualization | Matplotlib |
| Analytics | Scikit-Learn |
| Image Processing | PIL |
| Evaluation | Confusion Matrix |

---

# 📂 Project Structure

```bash
AgroVision-AI/
│
├── dataset/
│   └── PlantVillage/
│
├── models/
│   ├── best_model.pth
│   └── class_names.npy
│
├── outputs/
│   ├── accuracy.png
│   ├── loss.png
│   ├── confusion_matrix.png
│   └── prediction_chart.png
│
├── train.py
├── evaluate.py
├── predict.py
├── disease_info.py
│
├── requirements.txt
│
└── README.md
```

---

# 🚀 Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/AgroVision-AI.git

cd AgroVision-AI
```

## Create Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🏋️ Training

Train the model:

```bash
python train.py
```

The training pipeline:

- Loads dataset
- Applies augmentation
- Splits train/validation data
- Trains MobileNetV2
- Saves best model
- Generates performance graphs

---

# 📈 Training Results

## Performance Metrics

| Metric | Value |
|----------|--------|
| Training Accuracy | ~89.5% |
| Validation Accuracy | ~95% |
| Training Loss | ~0.31 |
| Validation Loss | ~0.17 |

---

## Accuracy Curve

The model demonstrates stable convergence with increasing validation accuracy throughout training.

✔ No severe overfitting

✔ Consistent learning behavior

✔ Strong generalization capability

---

# 🔬 Model Evaluation

Run evaluation:

```bash
python evaluate.py
```

Generated Outputs:

- Classification Report
- Accuracy Score
- Confusion Matrix
- Top Performing Classes

---

# 🎯 Disease Prediction

Run prediction:

```bash
python predict.py
```

The prediction engine:

1. Loads trained model
2. Processes input image
3. Generates probabilities
4. Predicts disease class
5. Calculates confidence score
6. Displays treatment information

---

# 📷 Sample Prediction

### Input

Tomato Leaf Image

### Prediction

```yaml
Disease:
Tomato Bacterial Spot

Confidence:
98.90%
```

---

### Top-5 Predictions

| Rank | Disease | Probability |
|--------|----------|-------------|
| 1 | Tomato Bacterial Spot | 98.90% |
| 2 | Tomato Yellow Leaf Curl Virus | 0.57% |
| 3 | Tomato Late Blight | 0.42% |
| 4 | Tomato Early Blight | 0.05% |
| 5 | Tomato Spider Mites | 0.02% |

---

# 💡 Key Features

### Smart Disease Diagnosis

Automatically identifies diseases from leaf images.

### Confidence-Based Predictions

Provides prediction certainty levels.

### Disease Knowledge Base

Displays disease causes and treatments.

### Visual Analytics

Includes:

- Accuracy Curves
- Loss Curves
- Confusion Matrix
- Probability Dashboard

### Scalable Architecture

Can be extended to:

- More crops
- More diseases
- Mobile applications
- Cloud deployment

---

# 📈 Business Impact

### Agriculture

- Early disease detection
- Reduced crop damage
- Improved productivity

### Economic

- Reduced monitoring costs
- Increased crop yield
- Better resource management

### Technological

- Precision agriculture
- AI-driven farming
- Smart crop monitoring

---

# 🔮 Future Roadmap

## Version 2.0

- Mobile Application
- Real-Time Camera Detection
- Drone-Based Monitoring
- IoT Sensor Integration
- Cloud Deployment
- Disease Severity Estimation
- Explainable AI (XAI)
- Multi-Language Support
- Farmer Chatbot Assistant
- Edge AI Deployment

---

# 📚 Research Contributions

This project demonstrates practical implementation of:

- Computer Vision
- Deep Learning
- Transfer Learning
- Agricultural AI
- Precision Farming
- Intelligent Decision Support Systems

---

# 👨‍💻 Author

### Sagnik Bachhar

**Data Science Intern**
**Pratinik Infotech**

📅 Internship Duration:
01 June 2026 – 31 July 2026

---

# 🏆 Internship Achievement

During this internship, a complete end-to-end Deep Learning pipeline was successfully developed for plant disease classification, achieving approximately **95% validation accuracy** while maintaining efficiency suitable for real-world agricultural deployment.

---

<div align="center">

## 🌱 "Empowering Agriculture Through Artificial Intelligence"

### If you found this project useful, consider giving it a ⭐

</div>

---

## Premium Additions (Demo, Visuals & Citation)

### 🎥 GIF Demo

An automated demo GIF has been generated from the training visuals. Open or embed it here:

![Live demo](assets/demo.gif)

### 🏗 Architecture Diagram

System architecture (vector):

![Architecture](assets/architecture.png)

### 📊 Visuals

- Accuracy: `assets/accuracy.png`
- Loss: `assets/loss.png`
- Confusion matrix: `assets/confusion_matrix.png`
- Prediction dashboard: `assets/prediction_dashboard.png`

### 🔢 Model Performance

Metric | Value
:--|:--
Training accuracy | 89.5%
Validation accuracy | 95.0%
Training loss | 0.31
Validation loss | 0.17
Number of classes | 15
Backbone | MobileNetV2 (ImageNet pretrained)

### 🔖 Badges (replace YOUR_GITHUB_USERNAME)

Add these at the top of the README to show repo metadata:

```
![GitHub stars](https://img.shields.io/github/stars/YOUR_GITHUB_USERNAME/AgroVision-AI?style=for-the-badge)
![GitHub forks](https://img.shields.io/github/forks/YOUR_GITHUB_USERNAME/AgroVision-AI?style=for-the-badge)
![GitHub last commit](https://img.shields.io/github/last-commit/YOUR_GITHUB_USERNAME/AgroVision-AI?style=for-the-badge)
![License](https://img.shields.io/github/license/YOUR_GITHUB_USERNAME/AgroVision-AI?style=for-the-badge)
```

### 📑 Citation & DOI

If you publish a release and archive it with Zenodo, add the DOI badge here. Example:

[![DOI](https://zenodo.org/badge/DOI/10.xxxx/zenodo.xxxxx.svg)](https://doi.org/10.xxxx/zenodo.xxxxx)

Replace `10.xxxx/zenodo.xxxxx` with your Zenodo DOI after publishing a release.

---

If you'd like, I can replace `YOUR_GITHUB_USERNAME` across the README and push the changes, or generate a higher-resolution PNG export of the `assets/architecture.png`.