import cv2
import mediapipe as mp
import numpy as np
import os
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd

recognized_text = ""  # Globalna promenljiva za prepoznati tekst

# za čuvanje slike cele reči 
def save_word_image(canvas, folder, idx):
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        
        word_img = canvas[y:y+h, x:x+w]
    else:
        word_img = canvas.copy()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(folder, f'word_{idx}_{timestamp}.png')
    success = cv2.imwrite(filename, word_img)
    print(f" Čuvam ceo crtež u: {filename}, success: {success}")
    if success:
        segment_letters(filename)
    return success

class EMNISTDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.X = df.iloc[:, 1:].values.reshape(-1, 28, 28).astype(np.uint8)
        self.y = df.iloc[:, 0].values.astype(np.int64) - 1  # oduzeto 1 da bi bilo od 0 do 25

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        image = self.X[idx]
        label = self.y[idx]
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0
        label = torch.tensor(label, dtype=torch.long)
        return image, label

# definicija CNN modela, da bi se koristila u handtrackingu direktno.
#
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.3)
        self.fcc1 = nn.Linear(128 * 3 * 3, 128)
        self.fcc2 = nn.Linear(128, 26)
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 3 * 3)
        x = F.relu(self.fcc1(x))
        x = self.dropout(x)
        x = self.fcc2(x)
        return x

# -IVO- ovo moras izmeniti za tvoje tezine ako ih budes imala, ili da ti posaljem moje koje sam korsitio ovde.
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'models', 'saved weights-labeled', '3. sa 3. transformisani dataset 1 model_epoch_9_20250630_125703.pth'
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cnn_model = CNN().to(device)
cnn_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
cnn_model.eval()


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

EMNIST_MAPPING_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'models', 'emnist-letters-mapping.txt'
)
EMNIST_LABELS_MAP = load_emnist_mapping(EMNIST_MAPPING_PATH)

#za cuvanje slova
last_segmented_letters = []  # lista tuple: (slika 28x28, predikcija)

def segment_letters(word_img_path):
    letters_dir = os.path.join(os.path.dirname(__file__), 'slova')
    os.makedirs(letters_dir, exist_ok=True)
    img = cv2.imread(word_img_path)
    if img is None:
        print(f"[DEBUG] Ne mogu da učitam sliku: {word_img_path}")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 15))
    morphed = cv2.dilate(thresh, kernel, iterations=1)
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(cnt) for cnt in contours]
    boxes.sort(key=lambda b: b[0])
    img_h, img_w = img.shape[:2]
    min_w, min_h = 10, 10
    max_w, max_h = int(0.9 * img_w), int(0.9 * img_h)
    letter_idx = 1
    recognized_letters = []
    global last_segmented_letters
    last_segmented_letters = []
    for (x, y, w, h) in boxes:
        if w < min_w or h < min_h:
            continue
        if w > max_w and h > max_h:
            continue
        pad = 5
        x1 = max(x - pad, 0)
        y1 = max(y - pad, 0)
        x2 = min(x + w + pad, img_w)
        y2 = min(y + h + pad, img_h)
        letter_img = img[y1:y2, x1:x2]
        letter_gray = cv2.cvtColor(letter_img, cv2.COLOR_BGR2GRAY)
        _, letter_bin = cv2.threshold(letter_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        letter_resized = cv2.resize(letter_bin, (28, 28), interpolation=cv2.INTER_AREA)
        letter_path = os.path.join(letters_dir, f"letter_{letter_idx}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png")
        cv2.imwrite(letter_path, letter_resized)
        print(f"[DEBUG] Sačuvano slovo: {letter_path}")
        # ovde cnn provaljuje slova
        try:
            tensor = torch.tensor(letter_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
            tensor = tensor.to(device)
            with torch.no_grad():
                output = cnn_model(tensor)
                pred_idx = int(torch.argmax(output, dim=1).item())
                emnist_label = pred_idx + 1  # EMNIST labels are 1-based
                pred_letter = EMNIST_LABELS_MAP.get(emnist_label, '?')
                recognized_letters.append(pred_letter)
                last_segmented_letters.append((letter_resized.copy(), pred_letter))
                print(f"[PREDICT] Prepoznato slovo: {pred_letter}")
        except Exception as e:
            print(f"[ERROR] Greška u prepoznavanju slova: {e}")
        letter_idx += 1
    if recognized_letters:
        word = ''.join(recognized_letters)
        print(f"[WORD] Prepoznata reč: {word}")
        global recognized_text
        if recognized_text:
            recognized_text += " "
        recognized_text += word

def draw_segmented_letters_panel(img, letters, panel_height=80, margin=10):
    if not letters:
        return
    n = len(letters)
    letter_size = 56
    panel_width = n * (letter_size + margin) + margin
    h, w = img.shape[:2]
    x0 = max((w - panel_width) // 2, 0)
    y0 = h - panel_height - 10
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_width, y0 + panel_height), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    for i, (letter_img, pred) in enumerate(letters):
        letter_img_color = cv2.cvtColor(letter_img, cv2.COLOR_GRAY2BGR)
        letter_img_resized = cv2.resize(letter_img_color, (letter_size, letter_size), interpolation=cv2.INTER_NEAREST)
        x = x0 + margin + i * (letter_size + margin)
        y = y0 + margin
        img[y:y+letter_size, x:x+letter_size] = letter_img_resized
        cv2.putText(img, str(pred), (x, y+letter_size+22), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2, cv2.LINE_AA)

def merge_letter_boxes(boxes, x_thresh=30, y_thresh=20):
    if not boxes:
        return []
    # Sortiraj po x
    boxes = sorted(boxes, key=lambda b: b[0])
    merged = []
    used = [False] * len(boxes)
    for i, (x, y, w, h) in enumerate(boxes):
        if used[i]:
            continue
        box = [x, y, w, h]
        for j in range(i+1, len(boxes)):
            x2, y2, w2, h2 = boxes[j]
            

            if abs((x + w) - x2) < x_thresh and (y < y2 + h2 and y + h > y2):
                
                nx = min(x, x2)
                ny = min(y, y2)
                nw = max(x + w, x2 + w2) - nx
                nh = max(y + h, y2 + h2) - ny
                box = [nx, ny, nw, nh]
                used[j] = True
        merged.append(tuple(box))
    return merged

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
canvas = None








data_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(data_dir, exist_ok=True)
word_count = 1  # Broj reči koje su sačuvane



def fingers_up(landmarks):
    tips = [4, 8, 12, 16, 20]  # indeksi vrhova prstiju: palac, kažiprst, srednji, domali, mali
    fingers = []
    # Palac (proverava da li je vrh desno od zgloba)
    fingers.append(landmarks[4].x > landmarks[3].x)
    # Ostali prsti (proverava da li je vrh iznad zgloba)
    for tip in tips[1:]:
        fingers.append(landmarks[tip].y < landmarks[tip - 2].y)
    return fingers  # [palac, kažiprst, srednji, domali, mali]

def draw_text_bubble(img, text, max_width=900, max_lines=4, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1.2, font_thickness=2, text_color=(30, 255, 30), bg_color=(30, 30, 30, 180), margin=20, line_spacing=12):
    
    words = text.split(' ')
    lines = []
    current_line = ''
    for word in words:
        test_line = current_line + (' ' if current_line else '') + word
        (w, h), _ = cv2.getTextSize(test_line, font, font_scale, font_thickness)
        if w > max_width and current_line:
            lines.append(current_line)
            current_line = word
        else:
            current_line = test_line
    if current_line:
        lines.append(current_line)
    # samo se poslednje prikazuju
    lines = lines[-max_lines:]
    # ukupna visina
    (w, h), _ = cv2.getTextSize('A', font, font_scale, font_thickness)
    total_height = h * len(lines) + line_spacing * (len(lines) - 1) + 2 * margin
    max_line_width = max([cv2.getTextSize(line, font, font_scale, font_thickness)[0][0] for line in lines] + [1])
    bubble_width = max_line_width + 2 * margin
    bubble_height = total_height
    # Pozicija teksta
    x = int((img.shape[1] - bubble_width) / 2)
    y = 10
    # Nacrtaj poluprovidnu pozadinu
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + bubble_width, y + bubble_height), bg_color[:3], -1)
    alpha = bg_color[3] / 255.0
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    # crtanje teksta
    y_text = y + margin + h
    for line in lines:
        (w_line, _), _ = cv2.getTextSize(line, font, font_scale, font_thickness)
        x_text = x + int((bubble_width - w_line) / 2)
        cv2.putText(img, line, (x_text, y_text), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        y_text += h + line_spacing

def draw_gesture_feedback(img, recognizing=False, eraser=False):
    h, w = img.shape[:2]
    overlay = img.copy()
    if recognizing:
        
        cv2.rectangle(overlay, (10, 10), (w-10, h-10), (0, 255, 0), 6)
        cv2.putText(overlay, "Samo malo", (w-230, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 3, cv2.LINE_AA)
    if eraser:
        )
        cv2.rectangle(overlay, (20, 20), (w-20, h-20), (0, 255, 255), 4)
        eraser_x, eraser_y = 60, 60
        cv2.rectangle(overlay, (eraser_x-25, eraser_y-15), (eraser_x+25, eraser_y+15), (0,255,255), -1, lineType=cv2.LINE_AA)
        cv2.ellipse(overlay, (eraser_x-25, eraser_y), (15,15), 0, 90, 270, (0,255,255), -1, lineType=cv2.LINE_AA)
        cv2.ellipse(overlay, (eraser_x+25, eraser_y), (15,15), 0, 270, 90, (0,255,255), -1, lineType=cv2.LINE_AA)
        
        cv2.rectangle(overlay, (eraser_x-25, eraser_y-15), (eraser_x+25, eraser_y+15), (80,80,80), 2, lineType=cv2.LINE_AA)
        cv2.ellipse(overlay, (eraser_x-25, eraser_y), (15,15), 0, 90, 270, (80,80,80), 2, lineType=cv2.LINE_AA)
        cv2.ellipse(overlay, (eraser_x+25, eraser_y), (15,15), 0, 270, 90, (80,80,80), 2, lineType=cv2.LINE_AA)
    alpha = 0.32 if recognizing or eraser else 0
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
) as hands:
    prev_point = None
    font = cv2.FONT_HERSHEY_SIMPLEX
    gesture_word_prev = False  # Da li je prethodni frejm bio gest za reč
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        if canvas is None or canvas.shape != frame.shape:
            canvas = np.zeros_like(frame)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        draw = False
        erase = False
        eraser_center = None
        gesture_word = False  # Da li je detektovan gest za reč
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                lm = hand_landmarks.landmark
                fingers = fingers_up(lm)
                print(f"[DEBUG] Fingers: {fingers}")
                # Ako je podignut samo kažiprst, crtamo
                if fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]:
                    draw = True
                    x = int(lm[8].x * frame.shape[1])
                    y = int(lm[8].y * frame.shape[0])
                    if prev_point is not None:
                        cv2.line(canvas, prev_point, (x, y), (0, 0, 255), 8)
                    prev_point = (x, y)
                # Ako su svi prsti (osim palca) podignuti, koristi se gumica
                elif all(fingers[1:]):
                    erase = True
                    eraser_center = (int(lm[0].x * frame.shape[1]), int(lm[0].y * frame.shape[0]))
                    prev_point = None
                else:
                    # Ako se prestane sa crtanjem, resetuj prev_point
                    prev_point = None
                # Ako su podignuti kažiprst i mali prst, a palac dole onda je     reč
                if fingers[1] and not fingers[0] and not fingers[2] and not fingers[3] and fingers[4]:
                    gesture_word = True
        else:
            prev_point = None
        # Sačuvaj
        if gesture_word and not gesture_word_prev:
            # Proveri da li na platnu ima crteža (da nije prazno)
            if np.any(canvas != 0):
                saved = save_word_image(canvas, data_dir, word_count)
                if saved:
                    word_count += 1
                    print(f"[DEBUG] Reč sačuvana, word_count: {word_count}")
                else:
                    print("[DEBUG] Reč nije sačuvana!")
                canvas = np.zeros_like(frame)
                prev_point = None
            else:
                print("[DEBUG] Preskočeno čuvanje prazne slike!")
        gesture_word_prev = gesture_word
        # gumica
        if erase and eraser_center is not None:
            cv2.circle(canvas, eraser_center, 40, (0, 0, 0), -1)
            cv2.circle(image, eraser_center, 40, (0, 255, 255), 2)
        
        img_out = cv2.addWeighted(image, 0.5, canvas, 0.5, 0)
        # Prikaz prepoznatog teksta na ekranu
        if recognized_text:
            draw_text_bubble(img_out, recognized_text)
        # Primer
        draw_gesture_feedback(img_out, recognizing=gesture_word, eraser=erase)
        if last_segmented_letters:
            draw_segmented_letters_panel(img_out, last_segmented_letters)
        cv2.imshow('MediaPipe Hands', img_out)

        if cv2.waitKey(1) & 0xFF == 27:
            break
        # Izađi ako je prozor zatvoren (klik na X)
        if cv2.getWindowProperty('MediaPipe Hands', cv2.WND_PROP_VISIBLE) < 1:
            break
cap.release()
cv2.destroyAllWindows()
