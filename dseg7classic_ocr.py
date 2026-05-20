import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional


# ─── Карта сегментов → цифра ────────────────────────────────────────────────
#
#   _
#  |_|   a=верх, b=верх-лево, c=верх-право
#  |_|   d=середина, e=низ-лево, f=низ-право, g=низ
#
# Сегменты: (a, b, c, d, e, f, g)


SEGMENT_MAP = {
    (1, 1, 1, 0, 1, 1, 1): 0,
    (0, 0, 1, 0, 0, 1, 0): 1,
    (1, 0, 1, 1, 1, 0, 1): 2,
    (1, 0, 1, 1, 0, 1, 1): 3,
    (0, 1, 1, 1, 0, 1, 0): 4,
    (1, 1, 0, 1, 0, 1, 1): 5,
    (1, 1, 0, 1, 1, 1, 1): 6,
    (1, 0, 1, 0, 0, 1, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 1, 0, 1, 1): 9,
}


@dataclass
class SegmentRegions:
    """Относительные зоны сегментов (0..1) для bbox цифры."""
    # (x_start, y_start, x_end, y_end) — нормализованные координаты
    a: tuple = (0.15, 0.00, 0.85, 0.12)  # верх
    b: tuple = (0.00, 0.05, 0.15, 0.48)  # верх-лево
    c: tuple = (0.85, 0.05, 1.00, 0.48)  # верх-право
    d: tuple = (0.15, 0.44, 0.85, 0.56)  # середина
    e: tuple = (0.00, 0.52, 0.15, 0.95)  # низ-лево
    f: tuple = (0.85, 0.52, 1.00, 0.95)  # низ-право
    g: tuple = (0.15, 0.88, 0.85, 1.00)  # низ


def preprocess(image: np.ndarray,
               threshold: int = 127,
               invert: bool = False) -> np.ndarray:
    """Бинаризация: серый + порог. invert=True если цифры тёмные на светлом."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    if invert:
        binary = cv2.bitwise_not(binary)
    return binary


def sample_segment(binary: np.ndarray,
                   bbox: tuple,
                   region: tuple,
                   fill_threshold: float = 0.25) -> int:
    """Проверяет, активен ли сегмент (доля белых пикселей > threshold)."""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    rx1 = int(x1 + region[0] * w)
    ry1 = int(y1 + region[1] * h)
    rx2 = int(x1 + region[2] * w)
    ry2 = int(y1 + region[3] * h)
    # Защита от нулевого ROI
    rx2 = max(rx2, rx1 + 1)
    ry2 = max(ry2, ry1 + 1)
    roi = binary[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return 0
    fill = np.count_nonzero(roi) / roi.size
    return 1 if fill > fill_threshold else 0


def recognize_digit(binary: np.ndarray,
                    bbox: tuple,
                    regions: SegmentRegions = SegmentRegions()) -> Optional[int]:
    """Распознаёт одну цифру по bbox (x1,y1,x2,y2)."""
    segments = tuple(
        sample_segment(binary, bbox, getattr(regions, seg))
        for seg in ('a', 'b', 'c', 'd', 'e', 'f', 'g')
    )
    return SEGMENT_MAP.get(segments)


def find_digit_bboxes(binary: np.ndarray,
                      min_area: int = 500,
                      aspect_ratio_range: tuple = (0.2, 1.0)) -> list:
    """Находит bounding boxes цифр через контуры."""
    # Морфология: закрываем мелкие дыры внутри цифры
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        ratio = w / h if h > 0 else 0
        if area >= min_area and aspect_ratio_range[0] <= ratio <= aspect_ratio_range[1]:
            bboxes.append((x, y, x + w, y + h))
    # Сортируем слева направо
    bboxes.sort(key=lambda b: b[0])
    return bboxes


def ocr_clock(image_path: str,
              threshold: int = 127,
              invert: bool = False,
              debug: bool = False) -> str:
    """
    Полный пайплайн: изображение → строка времени.

    Параметры:
        image_path  — путь к файлу
        threshold   — порог бинаризации (0-255)
        invert      — True если цифры тёмные на светлом фоне
        debug       — показать промежуточные изображения

    Возвращает строку, например '12:34' или '12?4' если цифра не распознана.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Не удалось открыть: {image_path}")

    binary = preprocess(img, threshold, invert)
    bboxes = find_digit_bboxes(binary)

    if debug:
        debug_img = img.copy()
        for bbox in bboxes:
            cv2.rectangle(debug_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.imwrite("debug_bboxes.png", debug_img)
        cv2.imwrite("debug_binary.png", binary)
        print(f"Найдено блоков: {len(bboxes)}")

    digits = []
    for bbox in bboxes:
        d = recognize_digit(binary, bbox)
        digits.append(str(d) if d is not None else '?')

    # Формируем строку времени: 4 цифры → HH:MM
    if len(digits) == 4:
        return f"{digits[0]}{digits[1]}:{digits[2]}{digits[3]}"
    elif len(digits) == 6:  # HH:MM:SS
        return f"{digits[0]}{digits[1]}:{digits[2]}{digits[3]}:{digits[4]}{digits[5]}"
    return ''.join(digits)


# ─── Использование ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = ocr_clock("clock_screenshot.png", threshold=100, debug=True)
    print(f"Время: {result}")
