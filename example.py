#!/usr/bin/env python3
"""
Пример программного использования генератора субтитров
"""

from subtitles import SubtitlesGenerator
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)

def main():
    # Создание генератора субтитров с моделью medium для более быстрой работы
    generator = SubtitlesGenerator(model_name="medium")
    
    # Путь к видеофайлу
    video_path = "path/to/your/video.mp4"
    
    # Генерация субтитров с расширенными параметрами
    output_files = generator.generate_subtitles(
        video_path=video_path,
        output_dir="output",  # Директория для сохранения результатов
        max_duration=300,     # Обработать только первые 5 минут видео
        languages=["ru", "en", "fr", "de"],  # Генерация субтитров на нескольких языках
        export_json=True      # Экспорт результатов в JSON
    )
    
    # Вывод информации о созданных файлах
    print("\nСозданные файлы:")
    for lang, path in output_files.items():
        print(f"- {lang}: {path}")

if __name__ == "__main__":
    main()
