import os 
import xml.etree.ElementTree as ET
import cv2
import numpy as np

def render_xml(xml_path, output_path, image_size=(800, 200)):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    image = np.ones((image_size[1], image_size[0]), dtype=np.uint8) * 255

    for stroke in root.iter('Stroke'):
        points = []
        for point in stroke:
            x, y = int(float(point.attrib['x'])), int(float(point.attrib['y']))
            points.append((x, y))
        for i in range(1, len(points)):
            cv2.line(image, points[i - 1], points[i], color=0, thickness=2)

    cv2.imwrite(output_path, image)

# === Putanje ===
input_dir = r"C:\vi\pfe\air_writting\iam_dataset\lineStrokes"
output_dir = r"C:\vi\pfe\air_writting\iam_dataset\line_images"

os.makedirs(output_dir, exist_ok=True)

# === Renderuj sve XML fajlove ===
for filename in os.listdir(input_dir):
    if filename.endswith(".xml"):
        xml_file = os.path.join(input_dir, filename)
        png_file = os.path.join(output_dir, filename.replace(".xml", ".png"))
        render_xml(xml_file, png_file)
