# tracking_utils.py
import numpy as np
from scipy.optimize import linear_sum_assignment

class ImprovedTracker:
    def __init__(self, max_age=10, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.next_id = 0
        self.max_tracks = 0
    
    def iou(self, box1, box2):
        """Вычисление Intersection over Union"""
        x1, y1, x2, y2 = box1
        x1_b, y1_b, x2_b, y2_b = box2
        
        # Вычисляем координаты пересечения
        xi1 = max(x1, x1_b)
        yi1 = max(y1, y1_b)
        xi2 = min(x2, x2_b)
        yi2 = min(y2, y2_b)
        
        # Площадь пересечения
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Площади bounding boxes
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_b - x1_b) * (y2_b - y1_b)
        
        # Площадь объединения
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def predict_current(self):
        """Возвращает текущие активные треки без обработки детекций"""
        # Увеличиваем возраст всех трекеров
        for tracker in self.trackers:
            tracker['age_since_update'] += 1
        
        # Удаляем старые трекеры
        self.trackers = [t for t in self.trackers if t['age_since_update'] < self.max_age]
        
        # Обновляем максимальное количество треков
        self.max_tracks = max(self.max_tracks, len(self.trackers))
        
        return self.trackers
    
    def update(self, detections):
        """Основной метод обновления трекеров с новыми детекциями"""
        self.frame_count += 1
        
        # Увеличиваем возраст всех трекеров
        for tracker in self.trackers:
            tracker['age_since_update'] += 1
            # Простое предсказание: предполагаем, что объект остался на месте
            tracker['predicted_bbox'] = tracker['bbox']
        
        # Если нет трекеров, создаем новые из детекций
        if len(self.trackers) == 0:
            for det in detections:
                self._create_new_tracker(det)
            return self.trackers
        
        # Если нет детекций, возвращаем текущие трекеры
        if len(detections) == 0:
            # Удаляем старые трекеры
            self.trackers = [t for t in self.trackers if t['age_since_update'] < self.max_age]
            return self.trackers
        
        # Матрица стоимости (IoU)
        cost_matrix = np.zeros((len(detections), len(self.trackers)))
        for d, det in enumerate(detections):
            for t, trk in enumerate(self.trackers):
                cost_matrix[d, t] = self.iou(det['bbox'], trk['bbox'])
        
        # Венгерский алгоритм для сопоставления
        row_ind, col_ind = linear_sum_assignment(-cost_matrix)
        
        matched_detections = set()
        matched_trackers = set()
        
        # Сопоставляем детекции с трекерами
        for d, t in zip(row_ind, col_ind):
            if cost_matrix[d, t] >= self.iou_threshold:
                self._update_tracker(self.trackers[t], detections[d])
                matched_detections.add(d)
                matched_trackers.add(t)
        
        # Создаем новые трекеры для несопоставленных детекций
        for d in range(len(detections)):
            if d not in matched_detections:
                self._create_new_tracker(detections[d])
        
        # Удаляем старые несопоставленные трекеры
        new_trackers = []
        for t, tracker in enumerate(self.trackers):
            if t in matched_trackers or tracker['age_since_update'] < self.max_age:
                new_trackers.append(tracker)
        self.trackers = new_trackers
        
        # Обновляем максимальное количество треков
        self.max_tracks = max(self.max_tracks, len(self.trackers))
        
        return self.trackers
    
    def _create_new_tracker(self, detection):
        """Создает новый трекер из детекции"""
        tracker = {
            'id': self.next_id,
            'category': detection['category'],
            'bbox': detection['bbox'],
            'positions': [detection['center']],
            'confidence': detection['confidence'],
            'age_since_update': 0,
            'hits': 1,
            'total_frames': 1
        }
        self.trackers.append(tracker)
        self.next_id += 1
    
    def _update_tracker(self, tracker, detection):
        """Обновляет существующий трекер новой детекцией"""
        tracker['bbox'] = detection['bbox']
        tracker['positions'].append(detection['center'])
        tracker['confidence'] = max(tracker['confidence'], detection['confidence'])
        tracker['age_since_update'] = 0  # Сбрасываем возраст
        tracker['hits'] += 1
        tracker['total_frames'] += 1
    
    def get_tracker_stats(self):
        """Возвращает статистику по трекерам"""
        active_trackers = [t for t in self.trackers if t['age_since_update'] == 0]
        return {
            'total_trackers': len(self.trackers),
            'active_trackers': len(active_trackers),
            'max_tracks': self.max_tracks,
            'next_id': self.next_id
        }