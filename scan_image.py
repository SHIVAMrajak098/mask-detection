from mtcnn import MTCNN
import cv2
import numpy as np
import sys
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Usage: python scan_image.py path/to/image.jpg
if len(sys.argv) < 2:
    print("Usage: python scan_image.py path/to/image.jpg")
    sys.exit(1)

image_path = sys.argv[1]

detector = MTCNN()
model = load_model('mask_detector_model.h5')
CLASSES = ['mask_weared_incorrect', 'with_mask', 'without_mask']
CONFIDENCE_THRESHOLD = 0.7
IMG_SIZE = 224

image = cv2.imread(image_path)
if image is None:
    print(f"Could not read image: {image_path}")
    sys.exit(1)

rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
faces = detector.detect_faces(rgb_image)

for face in faces:
    x, y, w, h = face['box']
    x, y = max(0, x), max(0, y)
    face_img = image[y:y+h, x:x+w]
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
    face_img = preprocess_input(face_img)
    face_img = np.expand_dims(face_img, axis=0)

    preds = model.predict(face_img)
    class_idx = np.argmax(preds[0])
    confidence = np.max(preds[0])
    if confidence > CONFIDENCE_THRESHOLD:
        label = CLASSES[class_idx]
    else:
        label = "uncertain"
    color = (0, 255, 0) if label == 'with_mask' else (0, 0, 255)

    print(f"Face at [{x}, {y}, {w}, {h}]: {label} (confidence: {confidence:.2f})")
    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

cv2.imshow('Result', image)
cv2.imwrite('scanned_result.png', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
print('Result saved as scanned_result.png') 