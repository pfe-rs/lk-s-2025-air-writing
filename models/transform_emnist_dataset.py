import pandas as pd
import numpy as np
import cv2
import os
import sys
#komanda za terminal kojom sam pokrenuo ovaj fajl, da bi mi se u istim folderima cuvali transformisane verije dataset-ova:
"""python3 models/transform_emnist_dataset.py /home/mihailo/Documents/projects/lk-s-2025-air-writing/dataset/emnist-letters-train.csv /home/mihailo/Documents/projects/lk-s-2025-air-writing/dataset/emnist-letters-train-transformed.csv
python3 models/transform_emnist_dataset.py /home/mihailo/Documents/projects/lk-s-2025-air-writing/models/emnist-letters-test.csv /home/mihailo/Documents/projects/lk-s-2025-air-writing/models/emnist-letters-test-transformed.csv"""

# slike iz dataset-a sam vracao na format koji dajem kada segmentiram slike iz reci.
def transform_image(img_arr):
    img = img_arr.reshape(28, 28).astype(np.uint8)
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img = cv2.flip(img, 0)
    return img.flatten()

def transform_csv(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    labels = df.iloc[:, 0].values
    images = df.iloc[:, 1:].values
    transformed_images = []
    for img in images:
        transformed = transform_image(img)
        transformed_images.append(transformed)
    transformed_images = np.array(transformed_images, dtype=np.uint8)
    out_df = pd.DataFrame(transformed_images)
    out_df.insert(0, 'label', labels)
    out_df.to_csv(output_csv, index=False)
    print(f"Transformisan dataset sačuvan u: {output_csv}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        
        print("Usage: python transform_emnist_dataset.py <input_csv> <output_csv>")
        sys.exit(1)
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    transform_csv(input_csv, output_csv)
