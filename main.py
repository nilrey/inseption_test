# video_object_detector.py
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
import cv2
import numpy as np
from collections import defaultdict
import os
from utils.tracking_utils import ImprovedTracker

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
        
        # Треккинг объектов
        self.track_history = defaultdict(lambda: [])
        self.next_object_id = 0
        self.tracked_objects = {}
        
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
    
    def track_objects(self, detections, max_distance=50):
        """Треккинг объектов между кадрами"""
        current_frame_objects = {}
        
        for detection in detections:
            center = detection['center']
            category = detection['category']
            confidence = detection['confidence']
            
            # Поиск ближайшего существующего объекта
            best_match_id = None
            min_distance = float('inf')
            
            for obj_id, obj_data in self.tracked_objects.items():
                if obj_data['category'] != category:
                    continue
                    
                # Вычисляем расстояние до последней позиции объекта
                last_center = obj_data['positions'][-1]
                distance = np.sqrt((center[0] - last_center[0])**2 + 
                                 (center[1] - last_center[1])**2)
                
                if distance < min_distance and distance < max_distance:
                    min_distance = distance
                    best_match_id = obj_id
            
            if best_match_id is not None:
                # Обновляем существующий объект
                obj_id = best_match_id
                self.tracked_objects[obj_id]['positions'].append(center)
                self.tracked_objects[obj_id]['bbox'] = detection['bbox']
                self.tracked_objects[obj_id]['confidence'] = confidence
                self.tracked_objects[obj_id]['active_frames'] += 1
            else:
                # Создаем новый объект
                obj_id = self.next_object_id
                self.next_object_id += 1
                self.tracked_objects[obj_id] = {
                    'category': category,
                    'positions': [center],
                    'bbox': detection['bbox'],
                    'confidence': confidence,
                    'active_frames': 1
                }
            
            current_frame_objects[obj_id] = self.tracked_objects[obj_id]
        
        # Удаляем объекты, которые не обновлялись
        inactive_objects = []
        for obj_id in self.tracked_objects:
            if obj_id not in current_frame_objects:
                inactive_objects.append(obj_id)
        
        for obj_id in inactive_objects:
            del self.tracked_objects[obj_id]
        
        return current_frame_objects
    
    def draw_detections(self, frame, tracked_objects):
        """Отрисовка bounding boxes и информации о треккинге"""
        for obj_id, obj_data in tracked_objects.items():
            category = obj_data['category']
            bbox = obj_data['bbox']
            confidence = obj_data['confidence']
            positions = obj_data['positions']
            
            # Цвет для категории
            color = self.colors.get(category, (255, 255, 255))
            
            # Рисуем bounding box
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Рисуем метку с ID и уверенностью
            label = f"{category} ID:{obj_id} ({confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Рисуем трек (историю перемещения)
            for i in range(1, len(positions)):
                if positions[i - 1] is None or positions[i] is None:
                    continue
                thickness = int(np.sqrt(64 / float(i + 1)) * 2)
                cv2.line(frame, positions[i - 1], positions[i], color, thickness)
                
    def process_video(self, input_video_path, output_video_path):
        """Основной метод обработки видео"""
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
        
        # Инициализируем переменную для хранения последних отслеживаемых объектов
        last_tracked_objects = {}
        
        print("Начало обработки видео...")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            
            # Обрабатываем каждый N-й кадр для увеличения производительности
            if frame_count % 3 == 0:  # Обрабатываем каждый 3-й кадр
                # Детекция объектов
                detections = self.sliding_window_detection(frame)
                
                # Треккинг объектов
                tracked_objects = self.track_objects(detections)
                
                # Сохраняем для использования в непроцессируемых кадрах
                last_tracked_objects = tracked_objects
                
                # Отрисовка результатов
                self.draw_detections(frame, tracked_objects)
                
                processed_frames += 1
            else:
                # Используем последние известные объекты для непроцессируемых кадров
                tracked_objects = last_tracked_objects
                # Отрисовываем объекты из предыдущего кадра
                self.draw_detections(frame, tracked_objects)
            
            # Добавляем информацию о кадре
            cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Objects: {len(tracked_objects)}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Записываем кадр в выходное видео
            out.write(frame)
            
            # Показываем прогресс
            if frame_count % 30 == 0:
                print(f"Обработано кадров: {frame_count}/{total_frames} "
                    f"({frame_count/total_frames*100:.1f}%)")
        
        # Освобождаем ресурсы
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"Обработка завершена!")
        print(f"Всего кадров: {frame_count}, обработано: {processed_frames}")
        print(f"Выходной файл: {output_video_path}")        


def main():
    # Инициализация детектора
    detector = VideoObjectDetector()
    
    # Пути к файлам
    input_video = "data/input/input_video.mp4"
    output_video = "data/output/output-inet-001.mp4"
    
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