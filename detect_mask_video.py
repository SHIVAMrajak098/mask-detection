from mtcnn import MTCNN
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load MTCNN face detector
face_detector = MTCNN()

# Load trained mask detector model
model = load_model('mask_detector_model.h5')

# Class labels (adjust if your dataset has different class names)
CLASSES = ['mask_weared_incorrect', 'with_mask', 'without_mask']
CONFIDENCE_THRESHOLD = 0.7
IMG_SIZE = 224

# Video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_detector.detect_faces(rgb_frame)

    for face in faces:
        x, y, w, h = face['box']
        x, y = max(0, x), max(0, y)
        face_img = frame[y:y+h, x:x+w]
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

        print(f"Predictions: {preds[0]}, Label: {label}, Confidence: {confidence:.2f}")
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    cv2.imshow('Face Mask Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 