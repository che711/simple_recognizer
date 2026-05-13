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

import sys
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
    return np.rot90(img, k=-k)          # rot90 вращает против часовой → инвертируем


def auto_rotate(img: np.ndarray, reader: easyocr.Reader) -> tuple[np.ndarray, int]:
    """
    Пробует 4 поворота (0, 90, 180, 270) и возвращает тот,
    при котором OCR нашёл паттерн времени HH:MM или HH:MM:SS.
    """
    time_pattern = re.compile(r'\d{1,2}[:\s]\d{2}')
    for angle in [0, 90, 270, 180]:
        rotated = rotate_image(img, angle)
        results = reader.readtext(rotated, detail=0)
        text = ' '.join(results)
        if time_pattern.search(text):
            return rotated, angle
    # Ничего не нашли — возвращаем оригинал
    return img, 0


def preprocess(img: np.ndarray) -> np.ndarray:
    """
    Конвертирует в серый, усиливает контраст, убирает шум.
    Хорошо работает для светлых символов на тёмном фоне (OLED).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE — адаптивное выравнивание гистограммы
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Небольшое размытие для удаления шума матрицы
    denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # Бинаризация (Otsu)
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary


# ─── Парсинг результатов OCR ───────────────────────────────────────────────────

def extract_time(ocr_results: list[str]) -> dict:
    """
    Ищет в списке строк OCR время и дату.
    Возвращает словарь {'time': '14:49:28', 'date': 'WE 12 MAY 2025', 'raw': [...]}.

    Логика выбора лучшего совпадения:
      1. Собираем ВСЕ кандидаты из всех строк.
      2. Отфильтровываем невалидные (час > 23, минуты > 59, секунды > 59).
      3. Предпочитаем полный формат HH:MM:SS перед HH:MM.
    """
    # Ищем паттерн вида  14:49:28  или  14.49.28  или  14 49 28
    # Требуем ровно 2 цифры для минут и секунд; для часов 1-2 цифры.
    time_re = re.compile(
        r'(?<!\d)'           # не предшествует цифра (границы слова)
        r'(\d{1,2})'         # часы
        r'[:.\ ]'            # разделитель
        r'(\d{2})'           # минуты
        r'(?:[:.\ ](\d{2}))?' # секунды (опционально)
        r'(?!\d)'            # не следует цифра
    )
    date_re = re.compile(
        r'(\w{2,3})\s+(\d{1,2})\s+(\w{3,9})\s+(\d{4})',
        re.IGNORECASE
    )

    candidates = []   # список (h, m, s, is_full)
    found_date = None

    for line in ocr_results:
        # Нормализуем: '.' и '-' → ':'
        normalized = re.sub(r'[.\-]', ':', line.strip())

        for m in time_re.finditer(normalized):
            h  = int(m.group(1))
            mi = int(m.group(2))
            s  = int(m.group(3)) if m.group(3) else 0
            is_full = m.group(3) is not None

            # Валидация диапазонов
            if 0 <= h <= 23 and 0 <= mi <= 59 and 0 <= s <= 59:
                candidates.append((h, mi, s, is_full))

        if not found_date:
            dm = date_re.search(line)
            if dm:
                found_date = ' '.join(dm.groups())

    # Выбираем лучший кандидат:
    # приоритет — полный формат (с секундами), затем первый валидный
    found_time = None
    if candidates:
        # сортируем: полные форматы (is_full=True) — первыми
        candidates.sort(key=lambda c: (not c[3],))
        h, mi, s, _ = candidates[0]
        found_time = f"{h:02d}:{mi:02d}:{s:02d}"

    return {
        'time': found_time,
        'date': found_date,
        'raw': ocr_results,
    }


# ─── Главная функция ───────────────────────────────────────────────────────────

def recognize(image_path: str, force_rotate: int | None = None, debug: bool = False) -> dict:
    """
    Основная функция распознавания.

    :param image_path:   путь к изображению
    :param force_rotate: принудительный поворот (0/90/180/270); None = авто
    :param debug:        сохранить предобработанные версии рядом с оригиналом
    :return:             словарь с 'time', 'date', 'raw', 'applied_rotation'
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Не удалось открыть файл: {image_path}")

    # Инициализируем EasyOCR (en — английские цифры и латиница)
    print("Инициализация EasyOCR…")
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)

    # Поворот
    if force_rotate is not None:
        img_rotated = rotate_image(img, force_rotate)
        applied_rotation = force_rotate
        print(f"Поворот: {force_rotate}° (задан вручную)")
    else:
        print("Авто-определение поворота…")
        img_rotated, applied_rotation = auto_rotate(img, reader)
        print(f"Поворот: {applied_rotation}°")

    # Предобработка
    preprocessed = preprocess(img_rotated)

    if debug:
        stem = Path(image_path).stem
        cv2.imwrite(f"{stem}_rotated.jpg",     img_rotated)
        cv2.imwrite(f"{stem}_preprocessed.jpg", preprocessed)
        print(f"[debug] сохранены: {stem}_rotated.jpg, {stem}_preprocessed.jpg")

    # OCR — пробуем сначала предобработанное, потом оригинал в цвете (как fallback)
    results_prep  = reader.readtext(preprocessed,  detail=0, paragraph=False)
    results_color = reader.readtext(img_rotated,   detail=0, paragraph=False)

    # Объединяем уникальные строки (предобработка приоритетна)
    all_lines = results_prep + [r for r in results_color if r not in results_prep]

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
    