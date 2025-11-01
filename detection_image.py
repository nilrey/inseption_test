# ready_to_use_detector.py
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np
import cv2

class ReadyPersonVehicleDetector:
    def __init__(self):
        print("Загрузка предобученной InceptionV3...")
        self.model = InceptionV3(weights='imagenet')
        self.input_size = (299, 299)
        
        # Классы ImageNet связанные с людьми и транспортом
        self.person_classes = ['person', 'man', 'woman', 'child', 'boy', 'girl']
        self.vehicle_classes = [
            'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'vehicle', 
            'ambulance', 'fire_engine', 'pickup', 'police_van', 'taxi'
        ]
    
    def predict_image(self, image_path):
        """Простое предсказание для одного изображения"""
        # Загрузка и предобработка
        img = tf.keras.utils.load_img(image_path, target_size=self.input_size)
        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Предсказание
        predictions = self.model.predict(img_array)
        decoded = tf.keras.applications.inception_v3.decode_predictions(predictions, top=10)[0]
        
        # Фильтрация результатов
        results = []
        for _, class_name, confidence in decoded:
            if class_name in self.person_classes:
                results.append(('person', class_name, confidence))
            elif class_name in self.vehicle_classes:
                results.append(('vehicle', class_name, confidence))
        
        return results

# Использование
if __name__ == "__main__":
    detector = ReadyPersonVehicleDetector()
    
    # Просто замените 'your_image.jpg' на путь к вашему изображению
    results = detector.predict_image('data/input/input_image.png')
    
    if results:
        print("Найдены объекты:")
        for category, specific_class, confidence in results:
            print(f"  {category} ({specific_class}): {confidence:.2%}")
    else:
        print("Люди или транспорт не обнаружены")