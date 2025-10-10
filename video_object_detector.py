# video_object_detector.py
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
import cv2
import numpy as np
from collections import defaultdict
import os
from tracking_utils import ImprovedTracker

class VideoObjectDetector:
    def __init__(self):
        print("Загрузка предобученной InceptionV3...")
        self.model = InceptionV3(weights='imagenet')
        self.input_size = (299, 299)
        
        # Классы для детекции
        self.target_classes = {
            'person': ['person', 'man', 'woman', 'child', 'boy', 'girl'],
            'vehicle': ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'vehicle', 
                       'ambulance', 'fire_engine', 'pickup', 'police_van', 'taxi',
                       'minivan', 'limousine', 'sports_car']
        }
        
        # Цвета для разных классов
        self.colors = {
            'person': (0, 255, 0),      # Зеленый
            'vehicle': (255, 0, 0)      # Синий
        }
        
        # Инициализация улучшенного трекера
        self.tracker = ImprovedTracker(max_age=10, min_hits=3, iou_threshold=0.3)
        
    def sliding_window_detection(self, frame, window_size=(299, 299), step_size=100):
        """Детекция объектов с помощью скользящего окна"""
        detections = []
        h, w = frame.shape[:2]
        
        for y in range(0, h - window_size[1], step_size):
            for x in range(0, w - window_size[0], step_size):
                # Вырезаем окно
                window = frame[y:y + window_size[1], x:x + window_size[0]]
                
                # Пропускаем маленькие или пустые окна
                if window.size == 0 or window.shape[0] < 50 or window.shape[1] < 50:
                    continue
                
                # Детекция в окне
                result = self.detect_in_window(window)
                if result:
                    category, confidence, class_name = result
                    # Сохраняем детекцию с координатами
                    detections.append({
                        'category': category,
                        'confidence': confidence,
                        'class_name': class_name,
                        'bbox': (x, y, x + window_size[0], y + window_size[1]),
                        'center': (x + window_size[0]//2, y + window_size[1]//2)
                    })
        
        return detections
    
    def detect_in_window(self, window):
        """Детекция объекта в одном окне"""
        try:
            # Предобработка
            window_rgb = cv2.cvtColor(window, cv2.COLOR_BGR2RGB)
            window_resized = cv2.resize(window_rgb, self.input_size)
            img_array = tf.keras.utils.img_to_array(window_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            # Предсказание
            predictions = self.model.predict(img_array, verbose=0)
            decoded = decode_predictions(predictions, top=3)[0]
            
            # Анализ результатов
            for _, class_name, confidence in decoded:
                for category, class_list in self.target_classes.items():
                    if class_name in class_list and confidence > 0.3:  # Порог уверенности
                        return category, confidence, class_name
        except Exception as e:
            pass
        
        return None
    
    def draw_detections_improved(self, frame, tracked_objects):
        """Отрисовка bounding boxes для улучшенного трекера"""
        for tracker in tracked_objects:
            obj_id = tracker['id']
            category = tracker['category']
            bbox = tracker['bbox']
            confidence = tracker['confidence']
            positions = tracker['positions']
            
            # Пропускаем объекты с малым количеством подтверждений
            if tracker['hits'] < self.tracker.min_hits:
                continue
            
            # Цвет для категории
            color = self.colors.get(category, (255, 255, 255))
            
            # Рисуем bounding box
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Рисуем метку с ID и уверенностью
            label = f"{category} ID:{obj_id} ({confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Фон для текста
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Рисуем трек (историю перемещения)
            for i in range(1, len(positions)):
                if positions[i - 1] is None or positions[i] is None:
                    continue
                # Толщина линии уменьшается для старых точек
                thickness = int(np.sqrt(64 / float(i + 1)) * 2)
                cv2.line(frame, positions[i - 1], positions[i], color, thickness)
            
            # Рисуем текущую позицию (точку)
            if positions:
                current_pos = positions[-1]
                cv2.circle(frame, current_pos, 5, color, -1)
    
    def process_video(self, input_video_path, output_video_path):
        """Основной метод обработки видео с улучшенным треккингом"""
        # Открываем входное видео
        cap = cv2.VideoCapture(input_video_path)
        
        if not cap.isOpened():
            print(f"Ошибка: Не удалось открыть видеофайл {input_video_path}")
            return
        
        # Получаем параметры видео
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Параметры видео: {width}x{height}, {fps} FPS, {total_frames} кадров")
        
        # Создаем VideoWriter для выходного видео
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        frame_count = 0
        processed_frames = 0
        
        print("Начало обработки видео с улучшенным треккингом...")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            
            # Обрабатываем каждый N-й кадр для увеличения производительности
            if frame_count % 3 == 0:  # Обрабатываем каждый 3-й кадр
                # Детекция объектов с помощью скользящего окна
                detections = self.sliding_window_detection(frame)
                
                # Улучшенный треккинг с ImprovedTracker
                tracked_objects = self.tracker.update(detections)
                
                # Отрисовка результатов с улучшенным треккингом
                self.draw_detections_improved(frame, tracked_objects)
                
                processed_frames += 1
            else:
                # Для непроцессорных кадров используем предсказания трекера
                tracked_objects = self.tracker.predict_current()
                self.draw_detections_improved(frame, tracked_objects)
            
            # Добавляем информацию о кадре
            cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Tracked Objects: {len(tracked_objects)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Tracker: ImprovedTracker", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Статистика трекера
            stats = self.tracker.get_tracker_stats()
            cv2.putText(frame, f"Total Tracks: {stats['max_tracks']}", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Записываем кадр в выходное видео
            out.write(frame)
            
            # Показываем прогресс
            if frame_count % 10 == 0:
                print(f"Обработано кадров: {frame_count}/{total_frames} "
                      f"({frame_count/total_frames*100:.1f}%)")
                print(f"Активных треков: {len(tracked_objects)}")
        
        # Освобождаем ресурсы
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Финальная статистика
        stats = self.tracker.get_tracker_stats()
        print(f"\nОбработка завершена!")
        print(f"Всего кадров: {frame_count}, обработано детекций: {processed_frames}")
        print(f"Максимальное количество треков: {stats['max_tracks']}")
        print(f"Всего создано треков: {stats['next_id']}")
        print(f"Выходной файл: {output_video_path}")

def main():
    # Инициализация детектора
    detector = VideoObjectDetector()
    
    # Пути к файлам
    input_video = "input_video.mp4"  # Замените на путь к вашему видео
    output_video = "output_video_with_improved_tracking.mp4"
    
    # Проверка существования входного файла
    if not os.path.exists(input_video):
        print(f"Ошибка: Входной файл {input_video} не найден!")
        print("Пожалуйста, укажите правильный путь к видеофайлу.")
        return
    
    # Обработка видео
    detector.process_video(input_video, output_video)
    
    print(f"Готово! Результат сохранен в: {output_video}")

if __name__ == "__main__":
    main()