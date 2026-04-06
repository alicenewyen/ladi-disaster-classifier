# 🌍 Satellite Image Damage Assessment (LADI-v2 + ResNet-18)

This project implements a deep learning-based multi-label classification system for post-disaster satellite imagery using the LADI-v2 dataset. The goal is to automatically identify damage-related features such as buildings, flooding, and debris from aerial images to support disaster response and assessment.

---

## 📌 Project Overview

Natural disasters often cause widespread damage that requires rapid assessment. Manual analysis of satellite images is time-consuming and inefficient. This project applies computer vision and deep learning techniques to automate damage detection from post-disaster imagery.

We use a ResNet-18 convolutional neural network (CNN) with transfer learning to classify multiple damage-related features present in each image.

---

## 🧠 Model Approach

- Architecture: ResNet-18 (pretrained on ImageNet)
- Task Type: Multi-label classification
- Loss Function: Binary Cross Entropy with Logits
- Output: Multiple labels per image (e.g., flooding + debris)

---

## 📊 Dataset

Dataset used: LADI-v2 (MIT Lincoln Laboratory Disaster Imagery Dataset)

### Dataset Features:
- ~10,000 post-disaster aerial images  
- Train / Validation / Test splits  
- Multi-label annotations including:
  - buildings_any
  - flooding_any
  - debris_any
  - roads_damage
  - trees_damage

Note: This dataset contains post-disaster images only (not pre/post pairs).

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/ladi-disaster-classifier.git
cd ladi-disaster-classifier
pip install -r requirements.txt