import cv2
import mediapipe as mp
import numpy as np
import os
import datetime
import sys
from pathlib import Path
from typing import List, Optional
import math
import json
import re
import time as _time
import random
import ctypes
from itertools import groupby
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
 & c:/Users/mihai/Documents/projects/lk-s-2025-air-writing/.venv_win/Scripts/python.exe c:/Users/mihai/Documents/projects/lk-s-2025-air-writing/Writing_part/hand_tracking.py
"""

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

try:
    from models.language_model.language_model import correct_word  # type: ignore
except ImportError:
    # Jezički model nije dostupan – radićemo bez korekcije.
    correct_word = None  # type: ignore

recognized_text = ""
recognized_history: List[str] = []
debug_letter_images: List[np.ndarray] = []
last_recognized_display: str = ""

# ── Game mode state ───────────────────────────────────────────────────────────
"""
GAME_WORDS = [
    "apple", "brave", "cloud", "dance", "eagle",
    "flame", "grace", "heart", "input", "joker",
    "knife", "lemon", "magic", "nerve", "ocean",
    "piano", "quiet", "river", "stone", "tiger",
    "ultra", "vivid", "water", "xenon", "yacht",
"""
GAME_WORDS = [
    "heart",
]
LEADERBOARD_PATH = CURRENT_DIR / "leaderboard.json"
LEADERBOARD_TOP_N = 5

game_mode: bool = False
# states: "nickname" | "ready" | "writing" | "result"
game_state: str = "nickname"
game_nickname: str = ""
game_nickname_input: str = ""       # buffer while user types
game_target_word: str = ""
game_start_time: float = 0.0
game_result_time: float = 0.0
game_result_recognized: str = ""
game_result_match: bool = False
game_thumbs_prev: bool = False
game_session_results: List[dict] = []   # results for current word/session


def load_leaderboard() -> List[dict]:
    if LEADERBOARD_PATH.exists():
        try:
            with open(LEADERBOARD_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def save_leaderboard(lb: List[dict]) -> None:
    with open(LEADERBOARD_PATH, "w") as f:
        json.dump(lb, f, indent=2)


def add_to_leaderboard(name: str, word: str, elapsed: float) -> List[dict]:
    lb = load_leaderboard()
    lb.append({"name": name, "word": word, "time": round(elapsed, 2)})
    # keep top 20 entries *per word* independently
    grouped: List[dict] = []
    for w, entries in groupby(sorted(lb, key=lambda e: e["word"]), key=lambda e: e["word"]):
        word_entries = sorted(entries, key=lambda e: e["time"])[:20]
        grouped.extend(word_entries)
    save_leaderboard(grouped)
    return grouped


def thumbs_up(lm) -> bool:
    """Thumb tip above thumb base, all other fingers curled."""
    thumb_up = lm[4].y < lm[2].y
    others_down = all(lm[tip].y > lm[tip - 2].y for tip in [8, 12, 16, 20])
    return thumb_up and others_down

# Normalizacija - iste vrednosti kao u treningu (models/data.py)
NORMALIZE_MEAN = 0.1736
NORMALIZE_STD = 0.3249

_user32 = ctypes.windll.user32
SCREEN_W: int = _user32.GetSystemMetrics(0)
SCREEN_H: int = _user32.GetSystemMetrics(1)

LETTER_GRID_COLS = 8
LETTER_TILE = 56
LETTER_MARGIN = 6
UI_PANEL_WIDTH = 380
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
    "Index finger      ->  Draw",
    "All fingers       ->  Erase",
    "Index + pinky     ->  Commit word",
    "ESC               ->  Exit",
]

# Dugme za brisanje
BUTTON_X = 16
BUTTON_Y = 0  # postavlja se dinamicki u build_ui_panel
BUTTON_W = 120
BUTTON_H = 32
BUTTON_COLOR = (60, 60, 180)
BUTTON_HOVER_COLOR = (80, 80, 220)
BUTTON_TEXT = "Clear All"
button_rect = {"x": 0, "y": 0, "w": BUTTON_W, "h": BUTTON_H}  # globalno za mouse callback
button_hovered = False

def normalize_text(text: str) -> str:
    return text.translate(ACCENT_MAP)


UI_INSTRUCTIONS = [normalize_text(line) for line in UI_INSTRUCTIONS_RAW]


def clear_all():
    """Obriši sve - canvas, prepoznati tekst, istoriju."""
    global recognized_text, recognized_history, debug_letter_images, last_recognized_display, canvas, word_count
    recognized_text = ""
    recognized_history.clear()
    debug_letter_images.clear()
    last_recognized_display = ""
    canvas = None
    word_count = 1
    print("[DEBUG] Sve obrisano!")


def mouse_callback(event, x, y, flags, param):
    """Callback za klik misa - proverava da li je kliknuto na dugme."""
    global button_hovered
    # inverse-transform from display space to dashboard space
    sc = _display_state["scale"]
    sx = int((x - _display_state["x_off"]) / sc)
    sy = int((y - _display_state["y_off"]) / sc)
    bx, by = button_rect["x"], button_rect["y"]
    bw, bh = button_rect["w"], button_rect["h"]
    button_hovered = bx <= sx <= bx + bw and by <= sy <= by + bh
    if event == cv2.EVENT_LBUTTONDOWN and button_hovered:
        clear_all()


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

_HAND_MODEL = Path(__file__).parent / "hand_landmarker.task"
if not _HAND_MODEL.exists():
    import urllib.request
    print("[INFO] Preuzimanje modela za detekciju ruke...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        _HAND_MODEL,
    )
    print("[INFO] Model preuzet.")

_HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17),
]

def _draw_landmarks(image, landmarks):
    h, w = image.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in _HAND_CONNECTIONS:
        cv2.line(image, pts[a], pts[b], (0, 255, 0), 2)
    for pt in pts:
        cv2.circle(image, pt, 4, (0, 0, 255), -1)

# Shared dict between mouse_callback and main loop for display scaling
_display_state = {"scale": 1.0, "x_off": 0, "y_off": 0}
# Pre-allocated fullscreen buffer (reused every frame to avoid GC pressure)
_padded_buf = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)

cap = cv2.VideoCapture(0)
canvas = None








data_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(data_dir, exist_ok=True)
word_count = 1  # Broj reči koje su sačuvane



def fingers_up(landmarks, is_right: bool = True):
    tips = [4, 8, 12, 16, 20]  # indeksi vrhova prstiju: palac, kažiprst, srednji, domali, mali
    fingers = []
    # Palac: smer zavisi od toga koja je ruka (nakon flipa slike)
    if is_right:
        fingers.append(landmarks[4].x > landmarks[3].x)
    else:
        fingers.append(landmarks[4].x < landmarks[3].x)
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
    video_width: int = 0,
) -> np.ndarray:
    panel = np.full((height, UI_PANEL_WIDTH, 3), UI_BG_COLOR, dtype=np.uint8)
    PAD = 20

    # ── Header bar ────────────────────────────────────────────────────────────
    HEADER_H = 46
    cv2.rectangle(panel, (0, 0), (UI_PANEL_WIDTH, HEADER_H), UI_ACCENT_COLOR, -1)
    hdr = "AIR WRITING SYSTEM"
    (hw, hh), _ = cv2.getTextSize(hdr, UI_FONT, 0.6, 2)
    cv2.putText(panel, hdr, ((UI_PANEL_WIDTH - hw) // 2, (HEADER_H + hh) // 2),
                UI_FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    y = HEADER_H + 18

    def separator(ypos: int) -> int:
        cv2.line(panel, (PAD, ypos), (UI_PANEL_WIDTH - PAD, ypos), (50, 50, 50), 1)
        return ypos + 14

    def section_label(text: str, ypos: int) -> int:
        label = normalize_text(text.upper())
        cv2.rectangle(panel, (PAD, ypos - 9), (PAD + 2, ypos + 3), UI_ACCENT_COLOR, -1)
        cv2.putText(panel, label, (PAD + 8, ypos),
                    UI_FONT, 0.42, UI_ACCENT_COLOR, 1, cv2.LINE_AA)
        return ypos + 14

    def body_lines(lines: List[str], ypos: int) -> int:
        for line in lines:
            cv2.putText(panel, normalize_text(line), (PAD + 4, ypos),
                        UI_FONT, UI_FONT_SCALE, UI_TEXT_COLOR, UI_FONT_THICKNESS, cv2.LINE_AA)
            ypos += 19
        return ypos

    # ── Status ────────────────────────────────────────────────────────────────
    y = section_label("Status", y)
    stats = [f"Words committed:  {max(0, word_count - 1)}"]
    if last_word:
        stats.append(f"Last word:  {last_word}")
    y = body_lines(stats, y)

    # ── Recognized text ───────────────────────────────────────────────────────
    y = separator(y + 8)
    y = section_label("Recognized Text", y)
    r_lines = wrap_text(normalize_text(recognized_text), UI_PANEL_WIDTH - PAD * 2 - 8)
    y = body_lines(r_lines if r_lines else ["--"], y)

    # ── Segmented letters ─────────────────────────────────────────────────────
    if letters_grid is not None:
        y = separator(y + 8)
        y = section_label("Segmented Letters", y)
        gmax_w = UI_PANEL_WIDTH - PAD * 2
        scale = min(1.0, gmax_w / letters_grid.shape[1])
        gw = int(letters_grid.shape[1] * scale)
        gh = int(letters_grid.shape[0] * scale)
        grid_r = cv2.resize(letters_grid, (gw, gh), interpolation=cv2.INTER_AREA)
        if y + gh > height - 155:
            y = max(height - 155 - gh, HEADER_H + 18)
        panel[y:y + gh, PAD:PAD + gw] = grid_r
        y += gh + 8

    # ── Controls ──────────────────────────────────────────────────────────────
    ctrl_y = max(y + 8, height - 168)
    ctrl_y = separator(ctrl_y)
    ctrl_y = section_label("Controls", ctrl_y)
    body_lines(UI_INSTRUCTIONS, ctrl_y)

    # ── Clear button ──────────────────────────────────────────────────────────
    btn_y = height - 46
    btn_x = PAD
    btn_w = UI_PANEL_WIDTH - PAD * 2
    btn_color = BUTTON_HOVER_COLOR if button_hovered else BUTTON_COLOR
    cv2.rectangle(panel, (btn_x, btn_y), (btn_x + btn_w, btn_y + BUTTON_H), btn_color, -1)
    cv2.rectangle(panel, (btn_x, btn_y), (btn_x + btn_w, btn_y + BUTTON_H), UI_ACCENT_COLOR, 1)
    lbl = normalize_text(BUTTON_TEXT)
    (lw, lh), _ = cv2.getTextSize(lbl, UI_FONT, 0.52, 1)
    cv2.putText(panel, lbl, (btn_x + (btn_w - lw) // 2, btn_y + (BUTTON_H + lh) // 2),
                UI_FONT, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    button_rect["x"] = video_width + btn_x
    button_rect["y"] = btn_y
    button_rect["w"] = btn_w
    button_rect["h"] = BUTTON_H

    return panel

def build_game_panel(height: int) -> np.ndarray:
    panel = np.full((height, UI_PANEL_WIDTH, 3), (18, 18, 18), dtype=np.uint8)
    PAD = 20

    # Header
    HEADER_H = 46
    cv2.rectangle(panel, (0, 0), (UI_PANEL_WIDTH, HEADER_H), (180, 80, 0), -1)
    hdr = "GAME MODE"
    (hw, hh), _ = cv2.getTextSize(hdr, UI_FONT, 0.6, 2)
    cv2.putText(panel, hdr, ((UI_PANEL_WIDTH - hw) // 2, (HEADER_H + hh) // 2),
                UI_FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    y = HEADER_H + 22

    def line(text, ypos, scale=0.5, color=(220, 220, 220), thickness=1):
        cv2.putText(panel, normalize_text(text), (PAD, ypos),
                    UI_FONT, scale, color, thickness, cv2.LINE_AA)
        return ypos + int(scale * 46) + 4

    def separator(ypos):
        cv2.line(panel, (PAD, ypos), (UI_PANEL_WIDTH - PAD, ypos), (50, 50, 50), 1)
        return ypos + 14

    if game_state == "nickname":
        y = line("Enter your nickname:", y, 0.52, (180, 180, 180))
        y += 6
        box_y = y
        cv2.rectangle(panel, (PAD, box_y), (UI_PANEL_WIDTH - PAD, box_y + 34), (40, 40, 40), -1)
        cv2.rectangle(panel, (PAD, box_y), (UI_PANEL_WIDTH - PAD, box_y + 34), (0, 180, 255), 1)
        display = game_nickname_input + "|"
        cv2.putText(panel, display, (PAD + 8, box_y + 23),
                    UI_FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y = box_y + 46
        y = line("Press ENTER to confirm.", y, 0.45, (120, 120, 120))

    elif game_state == "ready":
        y = line(f"Player:  {game_nickname}", y, 0.52, (0, 200, 120))
        y += 8
        y = separator(y)
        y = line("Write this word:", y, 0.52, (180, 180, 180))
        y += 4
        (ww, wh), _ = cv2.getTextSize(game_target_word.upper(), UI_FONT, 1.1, 2)
        cv2.putText(panel, game_target_word.upper(),
                    ((UI_PANEL_WIDTH - ww) // 2, y + wh + 4),
                    UI_FONT, 1.1, (0, 180, 255), 2, cv2.LINE_AA)
        y += wh + 22
        y = separator(y)
        y = line("Thumbs up  ->  Start timer", y, 0.47, (200, 200, 200))
        y = line("Index+pinky ->  Commit", y, 0.47, (200, 200, 200))

    elif game_state == "writing":
        elapsed = _time.time() - game_start_time
        y = line(f"Player:  {game_nickname}", y, 0.52, (0, 200, 120))
        y += 4
        y = separator(y)
        y = line("Write:", y, 0.5, (180, 180, 180))
        (ww, wh), _ = cv2.getTextSize(game_target_word.upper(), UI_FONT, 1.1, 2)
        cv2.putText(panel, game_target_word.upper(),
                    ((UI_PANEL_WIDTH - ww) // 2, y + wh + 4),
                    UI_FONT, 1.1, (0, 255, 120), 2, cv2.LINE_AA)
        y += wh + 20
        y = separator(y)
        # Stopwatch
        timer_str = f"{elapsed:.2f}s"
        (tw, th), _ = cv2.getTextSize(timer_str, UI_FONT, 1.3, 2)
        cv2.putText(panel, timer_str,
                    ((UI_PANEL_WIDTH - tw) // 2, y + th + 6),
                    UI_FONT, 1.3, (0, 180, 255), 2, cv2.LINE_AA)
        y += th + 22
        y = line("Index+pinky  ->  Commit", y, 0.45, (160, 160, 160))

    elif game_state == "result":
        y = line(f"Player:  {game_nickname}", y, 0.52, (0, 200, 120))
        y += 4
        y = separator(y)
        y = line(f"Target:      {game_target_word}", y, 0.5, (180, 180, 180))
        y = line(f"Recognized:  {game_result_recognized}", y, 0.5, (180, 180, 180))
        result_color = (0, 220, 80) if game_result_match else (0, 60, 220)
        result_label = "CORRECT!" if game_result_match else "WRONG"
        (rw, rh), _ = cv2.getTextSize(result_label, UI_FONT, 0.9, 2)
        cv2.putText(panel, result_label,
                    ((UI_PANEL_WIDTH - rw) // 2, y + rh + 6),
                    UI_FONT, 0.9, result_color, 2, cv2.LINE_AA)
        y += rh + 14
        if game_result_match:
            time_str = f"Time: {game_result_time:.2f}s"
            (tw, th), _ = cv2.getTextSize(time_str, UI_FONT, 0.8, 2)
            cv2.putText(panel, time_str,
                        ((UI_PANEL_WIDTH - tw) // 2, y + th + 4),
                        UI_FONT, 0.8, (0, 180, 255), 2, cv2.LINE_AA)
            y += th + 18
            y = separator(y)
            # Session standings — this round, same word
            cv2.putText(panel, f"THIS ROUND  [{game_target_word.upper()}]",
                        (PAD, y), UI_FONT, 0.45, (0, 180, 255), 1, cv2.LINE_AA)
            y += 18
            medals = ["1.", "2.", "3.", "4.", "5."]
            for i, entry in enumerate(game_session_results[:LEADERBOARD_TOP_N]):
                col = (0, 215, 100) if i == 0 else (180, 180, 180)
                txt = f"{medals[i]} {entry['name']:<12} {entry['time']:.2f}s"
                cv2.putText(panel, normalize_text(txt), (PAD + 4, y),
                            UI_FONT, 0.47, col, 1, cv2.LINE_AA)
                y += 18
            y = separator(y + 4)
            y = line("SPACE -> next player (same word)", y, 0.42, (140, 140, 140))
            y = line("N     -> new word (new session)", y, 0.42, (140, 140, 140))
            y = line("G     -> normal mode", y, 0.42, (140, 140, 140))
        else:
            y = separator(y)
            y = line("SPACE -> try again", y, 0.42, (140, 140, 140))
            y = line("N     -> next player (same word)", y, 0.42, (140, 140, 140))
            y = line("G     -> normal mode", y, 0.42, (140, 140, 140))

    # Footer hint
    cv2.putText(panel, "G = toggle normal mode", (PAD, height - 12),
                UI_FONT, 0.38, (70, 70, 70), 1, cv2.LINE_AA)

    return panel


_options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=str(_HAND_MODEL)),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
cv2.namedWindow('Air Writing System', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Air Writing System', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback('Air Writing System', mouse_callback)

with mp.tasks.vision.HandLandmarker.create_from_options(_options) as hands:
    prev_point = None
    gesture_word_prev = False
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        if canvas is None or canvas.shape != frame.shape:
            canvas = np.zeros_like(frame)

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        results = hands.detect_for_video(mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC)))
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        draw = False
        erase = False
        eraser_center = None
        gesture_word = False
        gesture_thumb = False

        if results.hand_landmarks:
            for i, lm in enumerate(results.hand_landmarks):
                _draw_landmarks(image, lm)
                # After cv2.flip, mediapipe "Left" = user's right hand, "Right" = user's left hand
                handedness_label = results.handedness[i][0].category_name if results.handedness else "Left"
                is_right = (handedness_label == "Left")
                fingers = fingers_up(lm, is_right)
                gesture_thumb = thumbs_up(lm)

                if fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]:
                    draw = True
                    x = int(lm[8].x * frame.shape[1])
                    y = int(lm[8].y * frame.shape[0])
                    if prev_point is not None:
                        cv2.line(canvas, prev_point, (x, y), (0, 0, 255), 8)
                    prev_point = (x, y)
                elif all(fingers[1:]):
                    erase = True
                    eraser_center = (int(lm[9].x * frame.shape[1]), int(lm[9].y * frame.shape[0]))
                    prev_point = None
                else:
                    prev_point = None

                if fingers[1] and not fingers[0] and not fingers[2] and not fingers[3] and fingers[4]:
                    gesture_word = True
        else:
            prev_point = None

        # ── Commit gesture (shared by normal + game writing state) ───────────
        if gesture_word and not gesture_word_prev:
            if np.any(canvas != 0):
                saved = save_word_image(canvas, data_dir, word_count)
                if saved:
                    word_count += 1
                    if game_mode and game_state == "writing":
                        game_result_time = _time.time() - game_start_time
                        # last_recognized_display is either "word" or "word (corrected)"
                        # prefer the corrected form for comparison
                        _disp = last_recognized_display if last_recognized_display else "?"
                        _m = re.search(r'\(([^)]+)\)', _disp)
                        game_result_recognized = _m.group(1) if _m else _disp.split()[0]
                        game_result_match = game_result_recognized.strip().lower() == game_target_word.strip().lower()
                        if game_result_match:
                            add_to_leaderboard(game_nickname, game_target_word, game_result_time)
                            game_session_results.append({"name": game_nickname, "time": game_result_time})
                            game_session_results.sort(key=lambda e: e["time"])
                        game_state = "result"
                canvas = np.zeros_like(frame)
                prev_point = None
        gesture_word_prev = gesture_word

        # ── Game mode: thumbs-up starts the timer ────────────────────────────
        if game_mode and game_state == "ready":
            if gesture_thumb and not game_thumbs_prev:
                game_start_time = _time.time()
                game_state = "writing"
                canvas = np.zeros_like(frame)
        game_thumbs_prev = gesture_thumb

        # ── Eraser ───────────────────────────────────────────────────────────
        if erase and eraser_center is not None:
            cv2.circle(canvas, eraser_center, 40, (0, 0, 0), -1)
            cv2.circle(image, eraser_center, 40, (0, 255, 255), 2)

        img_out = cv2.addWeighted(image, 0.5, canvas, 0.5, 0)

        # Gesture badge
        if draw:
            _badge, _badge_col = "DRAW", (0, 180, 255)
        elif erase:
            _badge, _badge_col = "ERASE", (0, 210, 80)
        elif gesture_word:
            _badge, _badge_col = "COMMIT", (255, 160, 0)
        elif game_mode and game_state == "ready" and gesture_thumb:
            _badge, _badge_col = "START", (0, 220, 80)
        else:
            _badge, _badge_col = "", (0, 0, 0)
        if _badge:
            (_bw, _bh), _ = cv2.getTextSize(_badge, UI_FONT, 0.62, 2)
            _bx, _by = 12, img_out.shape[0] - 14
            cv2.rectangle(img_out, (_bx - 8, _by - _bh - 8), (_bx + _bw + 8, _by + 6), (20, 20, 20), -1)
            cv2.rectangle(img_out, (_bx - 8, _by - _bh - 8), (_bx + _bw + 8, _by + 6), _badge_col, 1)
            cv2.putText(img_out, _badge, (_bx, _by), UI_FONT, 0.62, _badge_col, 2, cv2.LINE_AA)

        # ── Build side panel ─────────────────────────────────────────────────
        if game_mode:
            panel = build_game_panel(img_out.shape[0])
        else:
            letters_grid = build_letters_grid(debug_letter_images)
            panel = build_ui_panel(img_out.shape[0], recognized_text, last_recognized_display,
                                   letters_grid, word_count, img_out.shape[1])

        dashboard = np.hstack((img_out, panel))

        # Scale dashboard to fill screen, letterbox if aspect ratio differs
        dash_h, dash_w = dashboard.shape[:2]
        scale = min(SCREEN_W / dash_w, SCREEN_H / dash_h)
        disp_w = int(dash_w * scale)
        disp_h = int(dash_h * scale)
        x_off = (SCREEN_W - disp_w) // 2
        y_off = (SCREEN_H - disp_h) // 2
        _display_state["scale"] = scale
        _display_state["x_off"] = x_off
        _display_state["y_off"] = y_off
        _padded_buf[:] = 0
        _padded_buf[y_off:y_off + disp_h, x_off:x_off + disp_w] = cv2.resize(
            dashboard, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR
        )
        cv2.imshow('Air Writing System', _padded_buf)

        # ── Keyboard input ───────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('g') or key == ord('G'):
            game_mode = not game_mode
            if game_mode:
                # fresh session when entering game mode
                game_target_word = ""
                game_session_results.clear()
                game_state = "nickname"
                game_nickname = ""
                game_nickname_input = ""
                canvas = np.zeros_like(frame)
        elif game_mode and game_state == "nickname":
            if key == 13:  # ENTER
                if game_nickname_input.strip():
                    game_nickname = game_nickname_input.strip()
                    game_nickname_input = ""
                    # pick word only on the very first player of a session
                    if not game_target_word:
                        game_target_word = random.choice(GAME_WORDS)
                    game_state = "ready"
                    canvas = np.zeros_like(frame)
            elif key == 8:  # BACKSPACE
                game_nickname_input = game_nickname_input[:-1]
            elif 32 <= key <= 126:
                game_nickname_input += chr(key)
        elif game_mode and game_state == "result":
            if key == ord(' '):
                canvas = np.zeros_like(frame)
                debug_letter_images.clear()
                last_recognized_display = ""
                if game_result_match:
                    # correct — next player, same word
                    game_state = "nickname"
                    game_nickname = ""
                else:
                    # wrong — same player tries again
                    game_state = "ready"
            elif key == ord('n') or key == ord('N'):
                canvas = np.zeros_like(frame)
                debug_letter_images.clear()
                last_recognized_display = ""
                if game_result_match:
                    # correct — new word, new session
                    game_target_word = random.choice(GAME_WORDS)
                    game_session_results.clear()
                    game_state = "nickname"
                    game_nickname = ""
                else:
                    # wrong — next player, same word
                    game_state = "nickname"
                    game_nickname = ""

        if cv2.getWindowProperty('Air Writing System', cv2.WND_PROP_VISIBLE) < 1:
            break

cap.release()
cv2.destroyAllWindows()
