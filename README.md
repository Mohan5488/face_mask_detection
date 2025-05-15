## 😷 Face Mask Detection using CNN (From Scratch & Transfer Learning)

This project implements a Face Mask Detection system using both a custom-built Convolutional Neural Network (CNN) **from scratch** and **Transfer Learning** using MobileNetV2.

---

### 📂 Project Structure

The notebook is divided into two parts:

1. **From Scratch without Pretrained Model**
2. **Using Pretrained Model (MobileNetV2)**

---

### 🔍 Overview

This deep learning project:

* Detects if a person is wearing a mask or not.
* Uses images of people with and without masks.
* Implements CNN from scratch.
* Also uses Transfer Learning (MobileNetV2) for better accuracy and efficiency.

---

### 📁 Dataset

The dataset used is extracted from a ZIP archive stored in Google Drive.

Structure after extraction:

```
/content/FaceMask/data/
    ├── with_mask/
    └── without_mask/
```

Each folder contains relevant images for binary classification.

---

### 🚀 Features

* **CNN Architecture (from scratch)**:

  * Convolution, MaxPooling, Flatten, Dense, Dropout
* **Transfer Learning**:

  * MobileNetV2 base model
  * GlobalAveragePooling2D
  * Dense layers and dropout
  * Softmax activation for final classification

---

### ⚙️ Requirements

Install dependencies:

```bash
pip install tensorflow opencv-python matplotlib numpy
```

---

### 🛠️ How to Run

You can run the notebook on [Google Colab](https://colab.research.google.com/):

1. Upload dataset ZIP to Google Drive.
2. Mount drive in the notebook.
3. Execute the cells step by step.
4. Train model and evaluate accuracy.

---

### 🧱 Model Architecture Summary

#### 🔸 CNN From Scratch

* Input: (64x64x3) images
* Conv2D → MaxPool → Conv2D → MaxPool → Flatten → Dense → Output

#### 🔸 Transfer Learning (MobileNetV2)

* `include_top=False`
* `GlobalAveragePooling2D`
* Dense(128) → Dropout → Dense(2) with `softmax`

---

### 📊 Results

* Training and validation accuracy improves with MobileNetV2.
* From scratch model achieves decent results with limited data.
* MobileNetV2 provides faster and more accurate predictions.

---

### 📌 Notes

* Data augmentation or additional image generation techniques (e.g., `ImageDataGenerator`) may be used if the dataset is small.
* Use `expand_dims` to reshape input to 4D for prediction:

  ```python
  img = np.expand_dims(img, axis=0)
  ```

---

### 🧰 Tools & Libraries

* Python
* TensorFlow / Keras
* NumPy
* OpenCV
* Google Colab

---
