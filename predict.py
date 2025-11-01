# predict.py
import tensorflow as tf
import cv2
import numpy as np
import pickle
from PIL import Image
import os

class PersonVehicleDetector:
    def __init__(self, model_path='models/inception_model.h5'):
        self.model = tf.keras.models.load_model(model_path)
        self.input_size = (299, 299)
        
        # Загрузка названий классов
        with open('models/class_names.pkl', 'rb') as f:
            self.class_names = pickle.load(f)
    
    def preprocess_image(self, image):
        """Предобработка изображения для InceptionNet"""
        # Конвертация в RGB если нужно
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Изменение размера и нормализация
        image = cv2.resize(image, self.input_size)
        image = image.astype(np.float32) / 255.0
        
        return np.expand_dims(image, axis=0)
    
    def predict_image(self, image_path):
        """Предсказание для одного изображения"""
        # Загрузка изображения
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        
        # Предобработка
        processed_image = self.preprocess_image(image)
        
        # Предсказание
        predictions = self.model.predict(processed_image)
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx]
        
        return self.class_names[class_idx], confidence
    
    def predict_batch(self, image_paths):
        """Предсказание для списка изображений"""
        results = []
        for image_path in image_paths:
            try:
                class_name, confidence = self.predict_image(image_path)
                results.append({
                    'image_path': image_path,
                    'class': class_name,
                    'confidence': float(confidence)
                })
            except Exception as e:
                print(f"Ошибка при обработке {image_path}: {e}")
        
        return results
    
    def video_detection(self, input_path, output_path):
        """Реал-тайм детекция с веб-камеры"""
        cap = cv2.VideoCapture(input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        color_blue = (255, 0, 0)
        color_green = (0, 255, 0)
        color_red = (0, 0, 255)
        color_yellow = (0, 255, 255)
        
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Предсказание
            processed_frame = self.preprocess_image(frame)
            predictions = self.model.predict(processed_frame)
            class_idx = np.argmax(predictions[0])
            confidence = predictions[0][class_idx]
            
            # Отображение результата
            label = f"{self.class_names[class_idx]}: {confidence:.2f}"
            cv2.putText(frame, label, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            #cv2.imshow('Person/Vehicle Detection', frame)
            out.write(frame)
            
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    # Инициализация детектора
    detector = PersonVehicleDetector()
    
    input_video = "data/input/input_video.mp4"
    output_video = "data/output/output-inet-003.mp4"
    # Запуск реал-тайм детекции (раскомментировать при необходимости)
    detector.video_detection(input_video, output_video)

if __name__ == "__main__":
    main()