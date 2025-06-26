## 😷 Face Mask Detection System


[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-%E2%9D%A4-red)](https://keras.io/)
[![License](https://img.shields.io/badge/License-MIT-green)]()

A real-time face mask detection system that combines **MTCNN** for face detection with **MobileNetV2** for mask classification, built using modern deep learning frameworks.


## 📌 Overview
This system detects face masks using a combination of:

1. **MTCNN for face detection**

2. **MobileNetV2 for mask classification (3 classes)**

3. **Keras/TensorFlow for model training and inference**

It supports:

- Static image analysis

- Real-time webcam detection

- Training your own model with a labeled dataset

XML-based dataset parsing for face crops

## 📁 Project Structure
```


├── train.py                    # Trains the MobileNetV2 mask classifier
├── scan_image.py               # Runs detection on a given image
├── detectmaskedvideo.py       # Real-time mask detection via webcam
├── requirements.txt            # All dependencies for the project
├── image.png                   # Add image in the root folder to scan



## 🔧 Installation
First, make sure Python 3.7+ is installed.



1. **Clone the repository**

git clone https://github.com/SHIVAMrajak098/mask-detection.git
cd mask-detection

2. **Install dependencies**

pip install -r requirements.txt

3.**📦 Download Dataset**


Manually download the dataset from Kaggle:

👉 https://www.kaggle.com/datasets/andrewmvd/face-mask-detection

**Steps:**
- Sign in to Kaggle and open the link.

- Click "Download" to get the ZIP file.

- Extract it into a folder called dataset/ inside your project root.

## 🧼 Data Preprocessing (prepared_faces.py)
Before training, the dataset needs to be cleaned and prepared.
The raw Kaggle Face Mask Detection dataset provides Pascal VOC-style XML annotations that describe bounding boxes for different mask categories.

We use the script prepared_faces.py to:

🔍 Parse each XML annotation file

🖼️ Crop the face regions from the original images based on bounding box coordinates

📁 Organize and save them into class-wise folders inside the faces/ directory:

```
faces/
├── with_mask/
├── without_mask/
└── mask_weared_incorrect/ 
```
This ensures the model is trained directly on cropped face images, reducing noise and improving classification performance.

💡 This step is essential before running train.py, as it prepares the training data in the format required by Keras’ flow_from_directory.

## Train the Model
Train the MobileNetV2-based mask detector:
```
python train.py
```
- The trained model will be saved as `mask_detector_model.h5`.
- A training plot will be saved as `training_plot.png`.

## ⚙️ Training Details
Image Size: 224x224 (MobileNetV2 default)

**Augmentation:**

- Rotation (20°), Zoom (15%), Width/Height shift (20%)

- Shear, Horizontal flip, Fill mode: Nearest

- Validation Split: 20% (via ImageDataGenerator)

- Batch Size: 32

**Epochs:**

- Phase 1: 30 epochs (base model frozen)

- Phase 2: 10 epochs (last 20 layers unfrozen for fine-tuning)

##  🔧 Optimizer & Loss
**Optimizer:** Adam with learning_rate=1e-5

**Loss Function:** Categorical Crossentropy

**Callbacks:**

- EarlyStopping (patience=5)

- ModelCheckpoint (best model saved)



## Real-Time Mask Detection
Run the real-time detection script:
```bash
python detect_mask_video.py
```
- A webcam window will open.
- Detected faces will be labeled as `with_mask`, `without_mask`, or `mask_weared_incorrect`.
- Press `q` to quit.


## 🖼️ Run on Image

Run the script to scan images

```python scan_image.py path/to/image.jpg```

- Detects faces and classifies them

- Saves result as scanned_result.png


## 🔍 Classes Detected
✅ with_mask

❌ without_mask

⚠️ mask_weared_incorrect

uncertain (low confidence)

## Notes
- Make sure your webcam is connected and accessible.
- You can adjust parameters (batch size, epochs, etc.) in `train.py` as needed.
- The class labels in `detect_mask_video.py` should match your dataset's folder names.



## 🙋‍♂️ Author & Contact

> Developed with 💻 and ☕ by **[Shivam Rajak](https://github.com/SHIVAMrajak098)**

📫 **Contact me here:**

- GitHub: [github.com/SHIVAMrajak098](https://github.com/SHIVAMrajak098)
- LinkedIn: [linkedin.com/in/shivam-rajak](https://linkedin.com/in/shivam-rajak)
- Email: [shivamrajak098@gmail.com](mailto:shivamrajak098@gmail.com)

## Credits
- [Kaggle Face Mask Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)
- TensorFlow, Keras, OpenCV 

