"""
recognize_time.py
-----------------
Распознаёт время (и дату) с фотографий OLED/LCD дисплеев.
Умеет справляться с поворотом изображения, бликами и шумом.

Использование:
    python recognize_time.py image.jpg
    python recognize_time.py image.jpg --rotate 90
    python recognize_time.py image.jpg --debug
"""

import re
import argparse
from pathlib import Path

import cv2
import numpy as np
import easyocr


# ─── Предобработка ────────────────────────────────────────────────────────────

def rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    """Поворачивает изображение на заданный угол (кратный 90°)."""
    k = int(angle // 90) % 4
    return np.rot90(img, k=-k)


def get_preprocessed_variants(img: np.ndarray) -> list:
    """
    Возвращает несколько вариантов предобработки одного изображения.
    Разные варианты помогают, когда один из них не подходит для конкретного фото.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variants = []

    # 1. Просто серый
    variants.append(("gray", gray))

    # 2. CLAHE + Otsu (стандартный)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _, binary = cv2.threshold(
        cv2.GaussianBlur(enhanced, (3, 3), 0),
        0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    variants.append(("clahe_otsu", binary))

    # 3. Инвертированный (на случай тёмного текста на светлом фоне)
    variants.append(("clahe_otsu_inv", cv2.bitwise_not(binary)))

    # 4. Adaptive threshold (лучше при неравномерном освещении)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    adaptive = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10
    )
    variants.append(("adaptive", adaptive))

    # 5. Усиление резкости (помогает при размытых краях символов)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    variants.append(("sharp_gray", sharpened))

    return variants


# ─── Нормализация строк OCR ────────────────────────────────────────────────────

COLON_LOOKALIKES = str.maketrans({
    '.': ':',
    '-': ':',
    ';': ':',
    '|': ':',
})

def normalize_time_string(s: str) -> str:
    """Нормализует строку: заменяет визуально похожие на ':' символы."""
    return s.strip().translate(COLON_LOOKALIKES)


# ─── Сшивка разбитых строк ────────────────────────────────────────────────────

def stitch_split_time(lines: list) -> list:
    """
    EasyOCR иногда разбивает "10:51:44" на отдельные токены.
    Пробуем склеить соседние короткие числовые токены через ':'.
    """
    result = list(lines)
    digits_only = [re.sub(r'\D', '', l) for l in lines]

    i = 0
    while i < len(digits_only) - 1:
        parts = []
        j = i
        while j < len(digits_only) and 1 <= len(digits_only[j]) <= 2:
            parts.append(digits_only[j])
            j += 1
        if len(parts) >= 2:
            result.append(':'.join(parts))
        i += 1

    return result


# ─── Парсинг результатов OCR ───────────────────────────────────────────────────

def extract_time(ocr_results: list) -> dict:
    """
    Ищет в списке строк OCR время и дату.
    Возвращает словарь {'time': 'HH:MM:SS', 'date': '...', 'raw': [...]}.
    """
    time_re = re.compile(
        r'(?<!\d)'
        r'(\d{1,2})'
        r'[: ]'
        r'(\d{2})'
        r'(?:[: ](\d{2}))?'
        r'(?!\d)'
    )
    date_re = re.compile(
        r'(\w{2,3})\s+(\d{1,2})\s+(\w{3,9})\s+(\d{4})',
        re.IGNORECASE
    )

    candidates = []
    found_date = None

    extended = stitch_split_time(ocr_results)

    for line in extended:
        normalized = normalize_time_string(line)

        for m in time_re.finditer(normalized):
            h  = int(m.group(1))
            mi = int(m.group(2))
            s  = int(m.group(3)) if m.group(3) else None
            is_full = s is not None

            if 0 <= h <= 23 and 0 <= mi <= 59 and (s is None or 0 <= s <= 59):
                candidates.append((h, mi, s or 0, is_full))

        if not found_date:
            dm = date_re.search(line)
            if dm:
                found_date = ' '.join(dm.groups())

    found_time = None
    if candidates:
        candidates.sort(key=lambda c: (not c[3],))
        h, mi, s, _ = candidates[0]
        found_time = f"{h:02d}:{mi:02d}:{s:02d}"

    return {
        'time': found_time,
        'date': found_date,
        'raw': ocr_results,
    }


# ─── OCR по всем вариантам предобработки ──────────────────────────────────────

def _ocr_all_variants(img: np.ndarray, reader: easyocr.Reader) -> list:
    """
    Запускает OCR по всем вариантам предобработки.
    С allowlist='0123456789:. ' для числовых вариантов —
    резко снижает количество мусорных токенов.
    Плюс один проход без allowlist — для распознавания даты.
    """
    seen = set()
    all_lines = []

    for _, variant in get_preprocessed_variants(img):
        lines = reader.readtext(
            variant, detail=0, paragraph=False,
            allowlist='0123456789:. '
        )
        for l in lines:
            if l not in seen:
                seen.add(l)
                all_lines.append(l)

    # Проход без allowlist — для даты и любых нераспознанных символов
    for l in reader.readtext(img, detail=0, paragraph=False):
        if l not in seen:
            seen.add(l)
            all_lines.append(l)

    return all_lines


# ─── Auto-rotate ──────────────────────────────────────────────────────────────

def auto_rotate(img: np.ndarray, reader: easyocr.Reader) -> tuple:
    """
    Пробует 4 поворота и возвращает тот, при котором нашлось время.
    Предпочитает полный формат HH:MM:SS.
    """
    best_img, best_angle, best_is_full = img, 0, False

    for angle in [0, 90, 270, 180]:
        rotated = rotate_image(img, angle)
        lines = _ocr_all_variants(rotated, reader)
        parsed = extract_time(lines)

        if parsed['time']:
            is_full = parsed['time'].count(':') == 2
            print(f"  [{angle:3d}°] найдено: {parsed['time']}  raw: {lines}")
            if not best_is_full or (is_full and not best_is_full):
                best_img, best_angle, best_is_full = rotated, angle, is_full
            if is_full:
                break
        else:
            print(f"  [{angle:3d}°] не найдено  raw: {lines}")

    return best_img, best_angle


# ─── Главная функция ───────────────────────────────────────────────────────────

def recognize(image_path: str, force_rotate=None, debug: bool = False) -> dict:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Не удалось открыть файл: {image_path}")

    print("Инициализация EasyOCR…")
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)

    if force_rotate is not None:
        img_rotated = rotate_image(img, force_rotate)
        applied_rotation = force_rotate
        print(f"Поворот: {force_rotate}° (задан вручную)")
    else:
        print("Авто-определение поворота…")
        img_rotated, applied_rotation = auto_rotate(img, reader)
        print(f"Выбран поворот: {applied_rotation}°")

    if debug:
        stem = Path(image_path).stem
        cv2.imwrite(f"{stem}_rotated.jpg", img_rotated)
        for name, v in get_preprocessed_variants(img_rotated):
            cv2.imwrite(f"{stem}_{name}.jpg", v)
        print(f"[debug] сохранены варианты предобработки для {stem}")

    all_lines = _ocr_all_variants(img_rotated, reader)
    parsed = extract_time(all_lines)
    parsed['applied_rotation'] = applied_rotation
    return parsed


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Распознавание времени с фото дисплея")
    parser.add_argument("image", help="Путь к изображению")
    parser.add_argument("--rotate", type=int, choices=[0, 90, 180, 270],
                        default=None, help="Принудительный поворот в градусах")
    parser.add_argument("--debug", action="store_true",
                        help="Сохранить промежуточные изображения")
    args = parser.parse_args()

    result = recognize(args.image, force_rotate=args.rotate, debug=args.debug)

    print("\n─── Результат ───────────────────────────")
    print(f"  Время : {result['time']  or '(не найдено)'}")
    print(f"  Дата  : {result['date']  or '(не найдено)'}")
    print(f"  Поворот применён: {result['applied_rotation']}°")
    print(f"  Все строки OCR  : {result['raw']}")
    print("─────────────────────────────────────────")

    return result


if __name__ == "__main__":
    main()
    