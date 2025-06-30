import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import sys
import argparse
#ovaj fajl sam pokretao iz terminala jer nije htelo iz vscode.
#komanda : python /home/mihailo/Documents/projects/lk-s-2025-air-writing/models/test_emnist_dataset.py

TRAIN_CSV_PATH = "/home/mihailo/Documents/projects/lk-s-2025-air-writing/dataset/emnist-letters-train-transformed.csv"
TEST_CSV_PATH = "/home/mihailo/Documents/projects/lk-s-2025-air-writing/models/emnist-letters-test-transformed.csv"
MAPPING_PATH = "/home/mihailo/Documents/projects/lk-s-2025-air-writing/models/emnist-letters-mapping.txt"

# EMNIST mapping loader
def load_emnist_mapping(mapping_path):
    mapping = {}
    with open(mapping_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                idx = int(parts[0])
                letter = chr(int(parts[1]))  # ASCII code
                mapping[idx] = letter
    return mapping

def show_samples(csv_path, mapping_path, num_samples=8):
    df = pd.read_csv(csv_path)
    X = df.iloc[:, 1:].values.reshape(-1, 28, 28).astype(np.uint8)
    y = df.iloc[:, 0].values.astype(np.int64)
    mapping = load_emnist_mapping(mapping_path)
    indices = random.sample(range(len(y)), num_samples)
    plt.figure(figsize=(12, 2))
    for i, idx in enumerate(indices):
        img = X[idx]
        label = y[idx]
        letter = mapping.get(label, '?')
        plt.subplot(1, num_samples, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Label: {label}\nSlovo: {letter}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prikaz EMNIST primera iz train/test skupa.")
    parser.add_argument('--set', choices=['train', 'test'], default='train', help='Prikaz train ili test skupa')
    parser.add_argument('--num', type=int, default=8, help='Broj primera za prikaz')
    args = parser.parse_args()
    if args.set == 'train':
        show_samples(TRAIN_CSV_PATH, MAPPING_PATH, args.num)
    else:
        show_samples(TEST_CSV_PATH, MAPPING_PATH, args.num)
