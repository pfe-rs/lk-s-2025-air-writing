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
    bx, by = button_rect["x"], button_rect["y"]
    bw, bh = button_rect["w"], button_rect["h"]
    button_hovered = bx <= x <= bx + bw and by <= y <= by + bh
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

_options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=str(_HAND_MODEL)),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
cv2.namedWindow('Air Writing System')
cv2.setMouseCallback('Air Writing System', mouse_callback)

with mp.tasks.vision.HandLandmarker.create_from_options(_options) as hands:
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
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        results = hands.detect_for_video(mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC)))
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        draw = False
        erase = False
        eraser_center = None
        gesture_word = False  # Da li je detektovan gest za reč
        if results.hand_landmarks:
            for lm in results.hand_landmarks:
                _draw_landmarks(image, lm)
                fingers = fingers_up(lm)
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
                    eraser_center = (int(lm[9].x * frame.shape[1]), int(lm[9].y * frame.shape[0]))
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
        # Indikator aktivnog gesta (donji levi ugao kamere)
        if draw:
            _badge, _badge_col = "DRAW", (0, 180, 255)
        elif erase:
            _badge, _badge_col = "ERASE", (0, 210, 80)
        elif gesture_word:
            _badge, _badge_col = "COMMIT", (255, 160, 0)
        else:
            _badge, _badge_col = "", (0, 0, 0)
        if _badge:
            (_bw, _bh), _ = cv2.getTextSize(_badge, UI_FONT, 0.62, 2)
            _bx, _by = 12, img_out.shape[0] - 14
            cv2.rectangle(img_out, (_bx - 8, _by - _bh - 8), (_bx + _bw + 8, _by + 6), (20, 20, 20), -1)
            cv2.rectangle(img_out, (_bx - 8, _by - _bh - 8), (_bx + _bw + 8, _by + 6), _badge_col, 1)
            cv2.putText(img_out, _badge, (_bx, _by), UI_FONT, 0.62, _badge_col, 2, cv2.LINE_AA)
        letters_grid = build_letters_grid(debug_letter_images)
        panel = build_ui_panel(img_out.shape[0], recognized_text, last_recognized_display, letters_grid, word_count, img_out.shape[1])
        dashboard = np.hstack((img_out, panel))
        cv2.imshow('Air Writing System', dashboard)

        if cv2.waitKey(1) & 0xFF == 27:
            break
        # Izađi ako je prozor zatvoren (klik na X)
        if cv2.getWindowProperty('Air Writing System', cv2.WND_PROP_VISIBLE) < 1:
            break
cap.release()
cv2.destroyAllWindows()
