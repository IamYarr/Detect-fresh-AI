import os
import cv2
import json
import torch
import torch.nn.functional as F
from pathlib import Path

# Импортируем сборщик модели и классы/трансформации из наших модулей
from model import build_model
from dataset import val_transform, CLASSES

# 1. Настройка путей
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "Resources" / "models" / "mobilenetv3_freshness.pth"
TEST_DIR = BASE_DIR / "Resources" / "img" / "test"
RESULTS_DIR = BASE_DIR / "Resources" / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Настройки цветов для отрисовки в OpenCV (формат BGR)
COLORS = {
    "FRESH": (0, 255, 0),        # Зеленый
    "MINOR_DEFECT": (0, 255, 255), # Желтый
    "SPOILED": (0, 0, 255)       # Красный
}

def run_inference():
    # Проверка наличия обученной модели
    if not MODEL_PATH.exists():
        print(f"Ошибка: Веса модели не найдены по пути {MODEL_PATH}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Инференс запущен на: {device}")

    # Загружаем архитектуру и "надеваем" на неё сохраненные веса
    model = build_model(len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval() # Обязательно переводим в режим оценки

    results_data = []

    # Проходим по всем тестовым фото
    with torch.no_grad():
        for class_name in CLASSES:
            class_dir = TEST_DIR / class_name
            if not class_dir.exists():
                continue

            for img_path in class_dir.iterdir():
                if not img_path.is_file():
                    continue

                # 1. Чтение изображения для нейросети и для отрисовки
                img_bgr = cv2.imread(str(img_path))
                if img_bgr is None:
                    continue
                
                # Для нейросети нужен RGB формат
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                
                # Применяем валидационные трансформации (Resize + Normalize)
                transformed = val_transform(image=img_rgb)
                # Добавляем измерение батча (из [C, H, W] делаем [1, C, H, W])
                tensor = transformed['image'].unsqueeze(0).to(device)

                # 2. Прямой проход (получение сырых чисел - логитов)
                outputs = model(tensor)
                
                # 3. Превращаем логиты в проценты уверенности (от 0 до 1)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
                pred_class = CLASSES[predicted_idx.item()]
                conf_value = round(confidence.item(), 4)
                if conf_value < 0.65:
                    show_class=pred_class + " [MANUAL_REVIEW]"
                else: show_class=pred_class

                # Сохраняем в массив для JSON
                results_data.append({
                    "file_name": img_path.name,
                    "true_class": class_name,
                    "predicted_class": show_class,
                    "confidence": conf_value
                })

                text = f"{show_class}: {conf_value:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness = 2
                color = COLORS[pred_class]

                # Узнаем ширину исходной картинки
                img_height, img_width = img_bgr.shape[:2]
                
                # Черновой расчет размеров текста
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                
                # Максимально допустимая ширина текста (ширина фото минус 10px отступа с каждой стороны)
                max_text_width = img_width - 20
                
                # Если текст не влезает — пропорционально сжимаем шрифт
                if text_width > max_text_width:
                    font_scale = font_scale * (max_text_width / text_width)
                    # Если шрифт стал слишком мелким, убавляем толщину линий, чтобы текст не слипся в пятно
                    if font_scale < 0.5:
                        thickness = 1
                    # Чистовой перерасчет размеров с новым шрифтом
                    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

                # Рисуем плашку (она теперь всегда гарантированно поместится в кадр)
                cv2.rectangle(img_bgr, (10, 10), (10 + text_width, 10 + text_height + baseline), (0, 0, 0), cv2.FILLED)
                
                # Наносим текст
                cv2.putText(img_bgr, text, (10, 10 + text_height), font, font_scale, color, thickness)
                # 5. Сохраняем картинку с подписью
                save_path = RESULTS_DIR / f"demo_{img_path.name}"
                cv2.imwrite(str(save_path), img_bgr)

    # 6. Сохраняем JSON
    json_path = RESULTS_DIR / "inference_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=4, ensure_ascii=False)

    print(f"Готово! Обработано {len(results_data)} фото. Результаты в папке {RESULTS_DIR}")

if __name__ == "__main__":
    run_inference()
