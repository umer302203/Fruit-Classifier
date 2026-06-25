# 🍎 Fruit Classification Using Transfer Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange)
![Keras](https://img.shields.io/badge/Keras-2.15.0-red)
![Gradio](https://img.shields.io/badge/Gradio-4.31.5-green)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Spaces-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> **Real-time Fruit Classifier with 92% Accuracy — Deployed on Hugging Face Spaces**

<p align="center">
  <img src="https://img.shields.io/badge/Accuracy-92%25-brightgreen" alt="Accuracy">
  <img src="https://img.shields.io/badge/Categories-24-orange" alt="Categories">
  <img src="https://img.shields.io/badge/Model-VGG16-blue" alt="Model">
</p>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [Demo Video](#-demo-video)
- [Model Architecture](#-model-architecture)
- [Dataset](#-dataset)
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Results](#-results)
- [Challenges & Learnings](#-challenges--learnings)
- [Future Improvements](#-future-improvements)
- [Tech Stack](#-tech-stack)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

This project implements a **Fruit Classification System** using **Transfer Learning** with the **VGG16** architecture. The model is trained on the **Fruits-360** dataset and can classify **24 different fruit categories** with **92% test accuracy**.

The model is deployed as an interactive web application using **Gradio** and hosted on **Hugging Face Spaces** for real-time predictions.

---

## 🚀 Live Demo

<p align="center">
  <a href="https://huggingface.co/spaces/Umer78786/fruit-classifier">
    <img src="https://img.shields.io/badge/🚀%20Try%20Live%20Demo-Click%20Here-brightgreen?style=for-the-badge" alt="Live Demo">
  </a>
</p>

### 🌐 [Try the Fruit Classifier Here](https://huggingface.co/spaces/Umer78786/fruit-classifier)

---

## 🎥 Demo Video

A quick demonstration of the Fruit Classifier in action:

<p align="center">
  <video src="bandicam%202026-06-25%2013-54-40-865.mp4" controls width="80%">
    Your browser does not support the video tag.
  </video>
</p>

> **Note:** If the video doesn't play directly, download it from the repository:
> 📁 [`bandicam 2026-06-25 13-54-40-865.mp4`](bandicam%202026-06-25%2013-54-40-865.mp4)

---

## 🧠 Model Architecture

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:1400/1*BCpI83q90o7DwGVoymWgiw.jpeg" alt="VGG16 Architecture" width="80%">
</p>

The model is built using **Transfer Learning** with **VGG16** as the base architecture:

| Layer | Description |
|-------|-------------|
| **Base Model** | VGG16 (pre-trained on ImageNet) — `include_top=False` |
| **Pooling** | GlobalAveragePooling2D |
| **Dense Layer 1** | 256 units, ReLU activation |
| **Batch Normalization** | Normalizes activations for stable training |
| **Dropout** | 30% dropout for regularization |
| **Output Layer** | Dense (24 classes), Softmax activation |

### 📊 Model Summary

```
Total params: 14,979,368
Trainable params: 14,979,368
Non-trainable params: 0
Input shape: (64, 64, 3)
Output shape: (24,)
```

---

## 📂 Dataset

### Fruits-360 Dataset

- **Source:** [Fruits-360](https://www.kaggle.com/datasets/moltean/fruits)
- **Categories:** 24 fruit classes
- **Training Images:** ~24,000
- **Test Images:** ~6,000
- **Image Size:** 64x64 (resized from 100x100)

### Fruits Included:

| # | Fruit | # | Fruit |
|---|-------|---|-------|
| 1 | Apple (6 varieties) | 13 | Apple Braeburn |
| 2 | Apple Crimson Snow | 14 | Apple Golden |
| 3 | Apple Granny Smith | 15 | Apple Hit |
| 4 | Apple Pink Lady | 16 | Apple Red |
| 5 | Apple Red Delicious | 17 | Apple Red Yellow |
| 6 | Apple Rotten | 18 | Cabbage White |
| 7 | Carrot | 19 | Cucumber |
| 8 | Eggplant Violet | 20 | Pear |
| 9 | Zucchini | 21 | Zucchini Dark |
| 10 | ...and more! | | |

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔍 **Real-time Classification** | Upload an image and get instant predictions |
| 📊 **Confidence Scores** | See the model's confidence for each prediction |
| 🎯 **24 Categories** | Classifies a wide variety of fruits |
| 🖼️ **Clean UI** | User-friendly Gradio interface |
| 🌐 **Online Demo** | Accessible from anywhere via Hugging Face |
| 📱 **Mobile Responsive** | Works on mobile devices |

---

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- TensorFlow 2.15.0
- Git

### Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/fruit-classifier.git
cd fruit-classifier
```

### Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Requirements.txt

```
gradio==4.31.5
tensorflow==2.15.0
numpy==1.24.3
Pillow==10.2.0
matplotlib==3.7.1
```

---

## 💻 Usage

### Local Development

Run the application locally:

```bash
python app.py
```

### Make a Prediction

```python
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Load model
model = tf.keras.models.load_model('fruits_classifier.keras')

# Load class mapping
with open('class_indices.json', 'r') as f:
    class_names = json.load(f)

# Predict
def predict(image_path):
    img = Image.open(image_path).resize((64, 64))
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr)
    idx = np.argmax(preds)
    return class_names[str(idx)]
```

---

## 📁 Project Structure

```
fruit-classifier/
├── app.py                          # Gradio application
├── fruits_classifier.keras         # Trained model (root)
├── class_indices.json              # Class name mapping
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── LICENSE                         # MIT License
├── .gitignore                      # Git ignore file
├── bandicam 2026-06-25 13-54-40-865.mp4  # Demo video
├── notebooks/
│   └── training.ipynb              # Training notebook
└── models/
    └── fruits_classifier.keras     # Model (symlink/backup)
```

### 📁 Directory Details

| File/Directory | Description |
|----------------|-------------|
| `app.py` | Main Gradio web application |
| `fruits_classifier.keras` | Trained model (primary) |
| `class_indices.json` | Class name mapping (index → class) |
| `requirements.txt` | All Python dependencies |
| `README.md` | Project documentation |
| `LICENSE` | MIT License |
| `bandicam 2026-06-25 13-54-40-865.mp4` | Screen recording of the app |
| `notebooks/` | Jupyter notebooks for training |
| `models/` | Backup model location |

---

## 📊 Results

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **92.00%** |
| **Training Accuracy** | 94.5% |
| **Validation Accuracy** | 91.8% |
| **Loss** | 0.27 |

### Sample Predictions

<p align="center">
  <img src="https://i.imgur.com/placeholder-samples.png" alt="Sample Predictions" width="80%">
</p>

---

## 🧠 Challenges & Learnings

### Challenge 1: Preprocessing Mismatch

**Issue:** The model was trained with `rescale=1.0/255.0` in `ImageDataGenerator`, but deployment used `preprocess_input()` which applies mean subtraction.

**Solution:** Deploy with the same preprocessing as training:
```python
img_array = img_array / 255.0  # Simple rescaling
# Instead of preprocess_input()
```

### Challenge 2: Model Architecture Consistency

**Issue:** Rebuilding the model architecture manually for deployment risked mismatching the trained model.

**Solution:** Use `tf.keras.models.load_model()` to load the entire model file instead of rebuilding.

### Challenge 3: Class Name Formatting

**Issue:** Raw class names like `apple_red_1` were not user-friendly.

**Solution:** Added a `clean_class_name()` function to:
- Remove trailing numbers (`_1`, `_2`)
- Replace underscores with spaces
- Capitalize words

---

## 🚀 Future Improvements

| Improvement | Description |
|-------------|-------------|
| 📱 **Mobile App** | Build a React Native / Flutter mobile app |
| 🔄 **More Fruits** | Extend to 50+ fruit categories |
| 📊 **Grad-CAM** | Add visual explanations for predictions |
| 🗣️ **Voice Input** | Support for voice-based queries |
| 🌍 **Multi-language** | Support for multiple languages |

---

## 🛠️ Tech Stack

### Core
- **TensorFlow** — Deep Learning framework
- **Keras** — High-level API for neural networks
- **VGG16** — Pre-trained architecture for transfer learning

### Web Interface
- **Gradio** — Interactive ML app interface
- **Hugging Face Spaces** — Free deployment platform

### Data Processing
- **NumPy** — Numerical computing
- **Pandas** — Data manipulation
- **Pillow** — Image processing

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a new branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Contribution Guidelines

- Follow PEP 8 style guidelines
- Write meaningful commit messages
- Add tests for new features
- Update documentation accordingly

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 📞 Contact

<div align="center">

**Umer**  
[![GitHub](https://img.shields.io/badge/GitHub-Umer78786-black?style=flat&logo=github)](https://github.com/Umer78786)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://linkedin.com/in/umer78786)

</div>

---

## ⭐ Show Your Support

If you found this project helpful, please give it a ⭐ on GitHub and share it with others!

<p align="center">
  <a href="https://github.com/Umer78786/fruit-classifier">
    <img src="https://img.shields.io/github/stars/Umer78786/fruit-classifier?style=social" alt="GitHub stars">
  </a>
</p>

---

<p align="center">
  Made with ❤️ by <strong>Umer</strong>
</p>

---

### 📝 Notes

- The model file is stored both at the root (`fruits_classifier.keras`) and in the `models/` folder for backup purposes.
- The video demo (`bandicam 2026-06-25 13-54-40-865.mp4`) shows the Gradio interface in action.
- All preprocessing in deployment matches the training pipeline exactly for consistent predictions.

---

**Live Demo:** [https://huggingface.co/spaces/Umer78786/fruit-classifier](https://huggingface.co/spaces/Umer78786/fruit-classifier) 🚀
