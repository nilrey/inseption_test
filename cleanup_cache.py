# скрипт используется для очистки кеша кераса. была проблема с недокаченным файлом imagenet_class_index.json . Потребовалось очистить кеш и запустить основной скрипт video_object_detector.py заново.

import os
import shutil
from tensorflow.keras.utils import get_file

def clear_keras_cache():
    """Очищает кеш Keras и принудительно перезагружает файл"""
    # Путь к кешу Keras
    cache_dir = os.path.expanduser('~/.keras')
    
    # Удаляем файл, который скачался с ошибкой
    corrupted_file = os.path.join(cache_dir, 'models', 'imagenet_class_index.json')
    if os.path.exists(corrupted_file):
        os.remove(corrupted_file)
        print(f"Удален поврежденный файл: {corrupted_file}")
    
    # Также проверяем временные файлы
    temp_dir = os.path.join(cache_dir, 'datasets')
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            if 'imagenet_class_index' in file:
                os.remove(os.path.join(root, file))
                print(f"Удален временный файл: {file}")

if __name__ == "__main__":
    clear_keras_cache()
    print("Кеш очищен. Запустите основной скрипт снова.")