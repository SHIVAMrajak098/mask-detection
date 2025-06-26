import os
import cv2
import xml.etree.ElementTree as ET

ANNOTATIONS_DIR = os.path.join('dataset', 'annotations')
IMAGES_DIR = os.path.join('dataset', 'images')
OUTPUT_DIR = 'faces'

os.makedirs(OUTPUT_DIR, exist_ok=True)

for xml_file in os.listdir(ANNOTATIONS_DIR):
    if not xml_file.endswith('.xml'):
        continue
    tree = ET.parse(os.path.join(ANNOTATIONS_DIR, xml_file))
    root = tree.getroot()
    filename = root.find('filename').text
    image_path = os.path.join(IMAGES_DIR, filename)
    if not os.path.exists(image_path):
        continue
    image = cv2.imread(image_path)
    for obj in root.findall('object'):
        label = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        face = image[ymin:ymax, xmin:xmax]
        label_dir = os.path.join(OUTPUT_DIR, label)
        os.makedirs(label_dir, exist_ok=True)
        # Unique filename for each face
        base_name = os.path.splitext(filename)[0]
        face_filename = f"{base_name}_{xmin}_{ymin}_{xmax}_{ymax}.png"
        cv2.imwrite(os.path.join(label_dir, face_filename), face)
print('Face images have been prepared in the faces/ directory.') 