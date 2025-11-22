import cv2
import mediapipe as mp
import numpy as np
import os
import datetime
import sys
from pathlib import Path
from typing import List, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

try:
    from models.language_model.language_model import correct_word  # type: ignore
except ImportError:
    # Jezički model nije dostupan – radićemo bez korekcije.
    correct_word = None  # type: ignore

recognized_text = ""  # Globalna promenljiva za prikaz prepoznatog teksta
recognized_history: List[str] = []  # Čuvanje reči za potrebe LM-a kada je dostupan
debug_letter_images: List[np.ndarray] = []  # Slike svih slova poslednje segmentisane reči
last_recognized_display: str = ""

# Normalizacija - iste vrednosti kao u treningu (models/data.py)
NORMALIZE_MEAN = 0.1736
NORMALIZE_STD = 0.3249

LETTER_GRID_COLS = 8
LETTER_TILE = 56
LETTER_MARGIN = 6
UI_PANEL_WIDTH = 360
UI_BG_COLOR = (24, 24, 24)
UI_TEXT_COLOR = (230, 230, 230)
UI_ACCENT_COLOR = (0, 180, 255)
UI_SECTION_MARGIN = 18
UI_FONT = cv2.FONT_HERSHEY_SIMPLEX
UI_FONT_SCALE = 0.5
UI_FONT_THICKNESS = 1
ACCENT_MAP = str.maketrans({
    "č": "c", "ć": "c", "š": "s", "ž": "z", "đ": "dj",
    "Č": "C", "Ć": "C", "Š": "S", "Ž": "Z", "Đ": "DJ",
    "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u",
    "Á": "A", "É": "E", "Í": "I", "Ó": "O", "Ú": "U",
    "ä": "a", "ö": "o", "ü": "u", "ß": "ss",
})
UI_INSTRUCTIONS_RAW = [
    "Gest 1: Kažiprst = crtanje",
    "Gest 2: Svi prsti = gumica",
    "Gest 3: Kažiprst + mali = segment reči",
    "ESC: izlaz iz demonstracije",
]

def normalize_text(text: str) -> str:
    return text.translate(ACCENT_MAP)


UI_INSTRUCTIONS = [normalize_text(line) for line in UI_INSTRUCTIONS_RAW]


def resize_with_padding(img, size=28, margin=2):
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((size, size), dtype=img.dtype)
    max_dim = max(h, w)
    scale = (size - 2 * margin) / max_dim if max_dim > 0 else 1.0
    if scale <= 0:
        scale = 1.0
    scale = min(scale, 3.0)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
    canvas = np.zeros((size, size), dtype=img.dtype)
    x_offset = (size - new_w) // 2
    y_offset = (size - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas

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
    print(f"[DEBUG] Čuvam ceo crtež u: {filename}, success: {success}")
    if success:
        segment_letters(filename)
    return success

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

# Pokušavamo da pronađemo model automatski – prvo kroz env var, zatim iz direktorijuma.
def resolve_model_path() -> Path:
    env_path = os.environ.get("HANDTRACK_CNN_WEIGHTS")
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if path.exists():
            return path
        print(f"[WARN] HANDTRACK_CNN_WEIGHTS='{env_path}' ne postoji – koristim podrazumevani direktorijum.")

    weights_dir = PARENT_DIR / "models" / "saved weights-labeled"
    candidates = sorted(weights_dir.glob("*.pth"))
    if candidates:
        return candidates[-1]

    raise FileNotFoundError(
        "Nisu pronađene CNN težine. Postavi HANDTRACK_CNN_WEIGHTS ili ubaci .pth u models/saved weights-labeled."
    )


MODEL_PATH = resolve_model_path()

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
    display_letters: List[np.ndarray] = []
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
        letter_resized = resize_with_padding(letter_bin, size=28, margin=2)
        letter_path = os.path.join(letters_dir, f"letter_{letter_idx}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png")
        cv2.imwrite(letter_path, letter_resized)
        print(f"[DEBUG] Sačuvano slovo: {letter_path}")
        # čuvamo svaku sliku u BGR formatu za kasniji pregled
        letter_display = cv2.cvtColor(letter_resized, cv2.COLOR_GRAY2BGR)
        display_letters.append(letter_display)
        # ovde CNN prepoznaje slovo
        try:
            tensor = torch.tensor(letter_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
            tensor = (tensor - NORMALIZE_MEAN) / NORMALIZE_STD  # ista normalizacija kao u treningu
            tensor = tensor.to(device)
            with torch.no_grad():
                output = cnn_model(tensor)
                pred_idx = int(torch.argmax(output, dim=1).item())
                emnist_label = pred_idx + 1  # EMNIST labels are 1-based
                pred_letter = EMNIST_LABELS_MAP.get(emnist_label, '?')
                recognized_letters.append(pred_letter)
                print(f"[PREDICT] Prepoznato slovo: {pred_letter}")
        except Exception as e:
            print(f"[ERROR] Greška u prepoznavanju slova: {e}")
        letter_idx += 1
    if recognized_letters:
        word_raw = ''.join(recognized_letters)
        print(f"[WORD] Prepoznata reč: {word_raw}")
        global recognized_text, recognized_history, last_recognized_display
        debug_letter_images.clear()
        debug_letter_images.extend(display_letters)
        display_entry = word_raw
        if correct_word is not None:
            word_for_lm = word_raw.lower()
            try:
                corrected_word = correct_word(word_for_lm, recognized_history).lower()
            except Exception as lm_exc:
                print(f"[LANG] Greška pri korekciji: {lm_exc}")
                corrected_word = word_for_lm
            recognized_history.append(corrected_word)
            if corrected_word != word_for_lm:
                display_entry = f"{word_raw} ({corrected_word})"
                print(f"[LANG] Korigovana reč: {display_entry}")

        if recognized_text:
            recognized_text += " "
        recognized_text += display_entry
        last_recognized_display = display_entry


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


def build_letters_grid(letters: List[np.ndarray]) -> Optional[np.ndarray]:
    if not letters:
        return None
    cols = max(1, LETTER_GRID_COLS)
    rows = math.ceil(len(letters) / cols)
    tile = LETTER_TILE
    margin = LETTER_MARGIN
    grid_h = rows * tile + (rows + 1) * margin
    grid_w = cols * tile + (cols + 1) * margin
    grid = np.full((grid_h, grid_w, 3), 30, dtype=np.uint8)
    for idx, letter in enumerate(letters):
        r = idx // cols
        c = idx % cols
        y = margin + r * (tile + margin)
        x = margin + c * (tile + margin)
        resized = cv2.resize(letter, (tile, tile), interpolation=cv2.INTER_NEAREST)
        grid[y:y + tile, x:x + tile] = resized
    return grid


def wrap_text(text: str, max_width: int) -> List[str]:
    if not text:
        return []
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = (current + " " + word).strip()
        (w, _), _ = cv2.getTextSize(candidate, UI_FONT, UI_FONT_SCALE, UI_FONT_THICKNESS)
        if w > max_width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def build_ui_panel(
    height: int,
    recognized_text: str,
    last_word: str,
    letters_grid: Optional[np.ndarray],
    word_count: int,
) -> np.ndarray:
    panel = np.full((height, UI_PANEL_WIDTH, 3), UI_BG_COLOR, dtype=np.uint8)
    y = UI_SECTION_MARGIN + 5

    def draw_title(text: str, ypos: int, color=UI_ACCENT_COLOR):
        clean = normalize_text(text)
        cv2.putText(panel, clean, (16, ypos), UI_FONT, 0.7, color, 2, cv2.LINE_AA)
        return ypos + 8

    def draw_lines(lines: List[str], ypos: int):
        for line in lines:
            clean = normalize_text(line)
            cv2.putText(panel, clean, (16, ypos), UI_FONT, UI_FONT_SCALE, UI_TEXT_COLOR, UI_FONT_THICKNESS, cv2.LINE_AA)
            ypos += 20
        return ypos

    y = draw_title("Air Writing status", y)
    y += 12
    stats_lines = [
        normalize_text(f"Sačuvanih reči: {max(0, word_count - 1)}"),
    ]
    if last_word:
        stats_lines.append(normalize_text(f"Poslednja reč: {last_word}"))
    y = draw_lines(stats_lines, y)

    y += UI_SECTION_MARGIN
    y = draw_title("Prepoznati tekst", y)
    y += 12
    text_lines = wrap_text(normalize_text(recognized_text), UI_PANEL_WIDTH - 32)
    if not text_lines:
        text_lines = ["(čekam na unos)"]
    y = draw_lines(text_lines, y)

    if letters_grid is not None:
        y += UI_SECTION_MARGIN
        y = draw_title("Segmentisana slova", y)
        y += 12
        grid_max_width = UI_PANEL_WIDTH - 32
        scale = min(1.0, grid_max_width / letters_grid.shape[1])
        grid_resized = cv2.resize(letters_grid, (int(letters_grid.shape[1] * scale), int(letters_grid.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        h, w = grid_resized.shape[:2]
        if y + h + 10 > height - 100:
            y = height - 100 - h
            y = max(y, UI_SECTION_MARGIN)
        panel[y:y + h, 16:16 + w] = grid_resized
        y += h + 10

    y = max(y + 10, height - 90)
    y = draw_title("Kontrole", y)
    y += 12
    y = draw_lines(UI_INSTRUCTIONS, y)
    return panel

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
        # Gumica
        if erase and eraser_center is not None:
            cv2.circle(canvas, eraser_center, 40, (0, 0, 0), -1)
            cv2.circle(image, eraser_center, 40, (0, 255, 255), 2)
        # Kombinuj originalni snimak i platno da bi se videlo šta je nacrtano
        img_out = cv2.addWeighted(image, 0.5, canvas, 0.5, 0)
        # Prikaz prepoznatog teksta na ekranu
        if recognized_text:
            draw_text_bubble(img_out, recognized_text)
        letters_grid = build_letters_grid(debug_letter_images)
        panel = build_ui_panel(img_out.shape[0], recognized_text, last_recognized_display, letters_grid, word_count)
        dashboard = np.hstack((img_out, panel))
        cv2.imshow('Air Writing demonstrator', dashboard)

        if cv2.waitKey(1) & 0xFF == 27:
            break
        # Izađi ako je prozor zatvoren (klik na X)
        if cv2.getWindowProperty('Air Writing demonstrator', cv2.WND_PROP_VISIBLE) < 1:
            break
cap.release()
cv2.destroyAllWindows()
