# Air Writing Recognition

Real-time handwritten letter recognition using hand gestures captured via webcam. The system uses MediaPipe for hand tracking, a CNN trained on EMNIST Letters dataset for character recognition, and a language model for spell correction.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Training](#training)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## Installation

Requirements: Python 3.10+, webcam

```bash
git clone <repository-url>
cd lk-s-2025-air-writing

python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Usage

```bash
python Writing_part/hand_tracking.py
```

The application opens a window with camera feed and a UI panel on the right side.

### Gesture Controls

| Gesture | Action |
|---------|--------|
| Index finger only | Draw |
| All fingers raised | Erase |
| Index + pinky | Save word and trigger recognition |

Press **ESC** to exit.

### UI Panel

The right panel displays:
- Number of saved words
- Last recognized word (with correction if applicable)
- Segmented letters from the last word
- Gesture instructions

## How It Works

1. MediaPipe detects hand landmarks and tracks finger positions in real-time
2. Based on which fingers are raised, the system determines whether the user is drawing, erasing, or saving a word
3. When a word is saved, the image is segmented into individual letters using contour detection
4. Each letter is resized to 28x28 and passed through the CNN
5. The recognized word goes through a language model that uses Levenshtein distance and word frequency for correction

## Project Structure

```
Writing_part/hand_tracking.py       # main application
models/train_emnist_cnn.py          # CNN training script
models/data.py                      # dataset loading
models/saved weights-labeled/       # pre-trained weights (.pth)
models/language_model/              # spell correction module
config/language_layer.yaml          # language model config
```

## Training

To train the CNN from scratch:

1. Download EMNIST Letters dataset (CSV format) and place it in `dataset/`
2. Run training:

```bash
# Basic training
python models/train_emnist_cnn.py --epochs 30

# With GPU and mixed precision
python models/train_emnist_cnn.py --use-amp --epochs 30 --batch-size 256
```

The best model is saved to `checkpoints/best.pt`.

For detailed training options and output artifacts, see [README_emnist.md](README_emnist.md).

### Model Architecture

- 3 convolutional blocks (32 → 64 → 128 filters) with MaxPool2d
- Dropout (0.3) after first FC layer
- Output: 26 classes (A-Z)
- Training: AdamW optimizer, OneCycleLR scheduler, CrossEntropyLoss with label smoothing

Achieved accuracy on EMNIST test set: ~93%

## Configuration

Language model settings are in `config/language_layer.yaml`:

```yaml
language: en
max_edit_distance: 2
top_k_candidates: 5
distance_penalty: 1.0
```

## Troubleshooting

**Camera not opening**
- Check if the camera is connected and not used by another application
- Try changing camera index in `hand_tracking.py` line 277: `cv2.VideoCapture(1)` instead of `0`

**Model weights not found**
- The application looks for `.pth` files in `models/saved weights-labeled/`
- At least one checkpoint must be present

**Import errors**
- Make sure venv is activated and dependencies are installed from `requirements.txt`
