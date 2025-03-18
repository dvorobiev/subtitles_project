#!/usr/bin/env python3
"""
Subtitles Generator - Автоматический генератор субтитров для видео с поддержкой перевода
"""

# Информация о версии
__version__ = "1.0.1"
VERSION_INFO = {
    "major": 1,
    "minor": 0,
    "patch": 1,
    "release": "stable",
    "build_date": "2025-03-18"
}

import argparse
import os
import datetime
import srt
import logging
import time
import tempfile
from concurrent.futures import ThreadPoolExecutor
# Исправление импорта moviepy
import subprocess
from faster_whisper import WhisperModel
from tqdm import tqdm
import torch
import multiprocessing
from pathlib import Path
import shutil
import json

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class SubtitlesWriter:
    """Класс для обработки и записи субтитров в SRT формате."""
    
    def __init__(self, output_path, max_chars=80, min_segment_duration=0.5):
        """
        Инициализация объекта для записи субтитров.
        
        Args:
            output_path: Путь для сохранения файла субтитров
            max_chars: Максимальное количество символов в одной строке субтитров
            min_segment_duration: Минимальная длительность сегмента в секундах
        """
        self.output_path = output_path
        self.counter = 1
        self.buffer = []
        self.max_chars = max_chars
        self.min_segment_duration = min_segment_duration
        # Создаем директорию для файла, если она не существует
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Очищаем файл, если он уже существует
        if os.path.exists(output_path):
            os.remove(output_path)

    def split_text(self, text):
        """
        Разделяет длинный текст на части с учетом знаков препинания и максимальной длины.
        
        Args:
            text: Текст для разделения
            
        Returns:
            Список частей текста
        """
        if len(text) <= self.max_chars:
            return [text]
        
        parts = []
        current_part = ""
        
        # Приоритетные разделители
        separators = [". ", "! ", "? ", "; ", ", ", " "]
        
        words = text.split()
        for word in words:
            test_part = f"{current_part} {word}".strip()
            
            if len(test_part) <= self.max_chars:
                current_part = test_part
            else:
                if current_part:
                    parts.append(current_part)
                current_part = word
                
                # Если слово длиннее максимума, разбиваем его
                while len(current_part) > self.max_chars:
                    parts.append(current_part[:self.max_chars])
                    current_part = current_part[self.max_chars:]
        
        if current_part:
            parts.append(current_part)
            
        return parts

    def add_segment(self, segment):
        """
        Добавляет сегмент в буфер с учетом разделения длинных строк.
        
        Args:
            segment: Сегмент с текстом и временными метками
        """
        try:
            parts = self.split_text(segment.text.strip())
            segment_duration = segment.end - segment.start
            
            # Если сегмент слишком короткий, увеличиваем его длительность
            if segment_duration < self.min_segment_duration * len(parts):
                segment_duration = self.min_segment_duration * len(parts)
                
            duration_per_part = segment_duration / len(parts)
            
            for i, part in enumerate(parts):
                start_time = segment.start + (i * duration_per_part)
                end_time = start_time + duration_per_part
                
                sub = srt.Subtitle(
                    index=self.counter,
                    start=datetime.timedelta(seconds=start_time),
                    end=datetime.timedelta(seconds=end_time),
                    content=part
                )
                self.buffer.append(sub)
                self.counter += 1
                
                if len(self.buffer) >= 20:  # Увеличенный размер буфера для оптимизации I/O
                    self.flush()
        except Exception as e:
            logger.error(f"Ошибка добавления сегмента: {e}")

    def flush(self):
        """Записывает содержимое буфера в файл и очищает буфер."""
        if not self.buffer:
            return
            
        try:
            with open(self.output_path, "a", encoding="utf-8") as f:
                f.write(srt.compose(self.buffer))
            self.buffer = []
        except Exception as e:
            logger.error(f"Ошибка записи в файл: {e}")
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()


class SubtitlesGenerator:
    """Основной класс для генерации субтитров из видео."""
    
    def __init__(self, model_name="large-v3", device=None, compute_type=None):
        """
        Инициализация генератора субтитров.
        
        Args:
            model_name: Название модели Whisper для использования
            device: Устройство для вычислений (cuda, mps, cpu)
            compute_type: Тип вычислений (float16, int8)
        """
        self.version = __version__
        
        # Определение оптимального устройства, если не указано
        if device is None or compute_type is None:
            device, compute_type = self.get_optimal_device()
            
        logger.info(f"Subtitles Generator v{self.version}")
        logger.info(f"Используется устройство: {device}, тип вычислений: {compute_type}")
        
        # Загрузка модели
        logger.info(f"Загрузка модели {model_name}...")
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)
        
        # Оптимизация многопоточности
        os.environ["OMP_NUM_THREADS"] = str(max(1, multiprocessing.cpu_count() - 1))
        
        # Создание временной директории для промежуточных файлов
        self.temp_dir = tempfile.mkdtemp(prefix="subtitles_gen_")
        logger.debug(f"Создана временная директория: {self.temp_dir}")

    def get_optimal_device(self):
        """
        Определяет оптимальное устройство для работы модели.
        
        Returns:
            Кортеж (устройство, тип вычислений)
        """
        if torch.cuda.is_available():
            return "cuda", "float16"
        # MPS не поддерживается faster-whisper, используем CPU
        return "cpu", "int8"

    def extract_audio(self, video_path, max_duration=None):
        """
        Извлекает аудио из видеофайла с оптимизированными параметрами.
        
        Args:
            video_path: Путь к видеофайлу
            max_duration: Максимальная длительность обработки в секундах
            
        Returns:
            Кортеж (путь к аудиофайлу, длительность)
        """
        logger.info("Начало извлечения аудио...")
        audio_path = os.path.join(self.temp_dir, "audio_extract.wav")
        
        try:
            # Извлечение аудио с помощью ffmpeg
            subprocess.run([
                "ffmpeg",
                "-i", video_path,
                "-vn",  # Без видео
                "-ar", "16000",  # Частота дискретизации
                "-ac", "1",  # Моно
                "-threads", str(max(1, multiprocessing.cpu_count() - 1)),  # Оптимальное количество потоков
                audio_path
            ], check=True)
            
            # Определение длительности аудио
            duration = subprocess.run([
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                audio_path
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.decode().strip()
            duration = float(duration)
            
            if max_duration is not None:
                duration = min(duration, max_duration)
                
            logger.info(f"Длительность обработки: {duration:.2f} сек")
                
            logger.info(f"Аудио сохранено в: {audio_path}")
            return audio_path, duration
        except Exception as e:
            logger.error(f"Ошибка при извлечении аудио: {e}")
            raise

    def process_audio(self, audio_path, total_duration, output_srt, task, language="ru"):
        """
        Выполняет транскрипцию или перевод аудио с улучшенной обработкой сегментов.
        
        Args:
            audio_path: Путь к аудиофайлу
            total_duration: Общая длительность аудио
            output_srt: Путь для сохранения субтитров
            task: Задача (transcribe или translate)
            language: Язык аудио
            
        Returns:
            Список сегментов
        """
        logger.info(f"Начало {'транскрипции' if task=='transcribe' else 'перевода'} аудио...")
        
        try:
            segments, _ = self.model.transcribe(
                audio_path,
                language=language,
                task=task,
                beam_size=5,
                vad_filter=True,
                word_timestamps=True
            )

            with SubtitlesWriter(output_srt) as writer:
                with tqdm(
                    total=total_duration,
                    desc="Обработка аудио",
                    unit="сек",
                    bar_format="{l_bar}{bar}| {n:.1f}/{total_fmt} сек [{elapsed}<{remaining}]"
                ) as pbar:
                    last_end = 0
                    for segment in segments:
                        writer.add_segment(segment)
                        progress = max(0, segment.end - last_end)
                        pbar.update(progress)
                        last_end = segment.end
                        pbar.set_postfix_str(f"Фраз: {writer.counter - 1}")
                        
            return segments
        except Exception as e:
            logger.error(f"Ошибка при обработке аудио: {e}")
            raise

    def generate_subtitles(self, video_path, output_dir=None, max_duration=None, 
                          languages=None, export_json=False):
        """
        Генерирует субтитры для видеофайла.
        
        Args:
            video_path: Путь к видеофайлу
            output_dir: Директория для сохранения результатов
            max_duration: Максимальная длительность обработки в секундах
            languages: Список языков для перевода (по умолчанию только русский и английский)
            export_json: Экспортировать ли результаты в JSON формате
            
        Returns:
            Словарь с путями к созданным файлам
        """
        try:
            # Проверка входного файла
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Видеофайл не найден: {video_path}")
                
            # Определение выходной директории
            if output_dir is None:
                output_dir = os.path.dirname(os.path.abspath(video_path))
            os.makedirs(output_dir, exist_ok=True)
            
            # Определение языков
            if languages is None:
                languages = ["ru", "en"]
            elif isinstance(languages, str):
                languages = [languages]
                
            logger.info(f"Начало обработки файла: {video_path}")
            audio_path, duration = self.extract_audio(video_path, max_duration)
            
            # Формирование путей для выходных файлов
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_files = {}
            
            # Транскрипция на исходном языке (русский)
            output_srt_ru = os.path.join(output_dir, f"{base_name}_subs.srt")
            segments_ru = self.process_audio(audio_path, duration, output_srt_ru, task="transcribe", language="ru")
            output_files["ru"] = output_srt_ru
            logger.info(f"Русские субтитры сохранены в: {output_srt_ru}")
            
            # Перевод на другие языки
            for lang in languages:
                if lang != "ru":
                    output_srt_lang = os.path.join(output_dir, f"{base_name}_subs_{lang}.srt")
                    self.process_audio(audio_path, duration, output_srt_lang, task="translate", language=lang)
                    output_files[lang] = output_srt_lang
                    logger.info(f"Перевод субтитров на {lang} сохранен в: {output_srt_lang}")
            
            # Экспорт в JSON, если требуется
            if export_json:
                json_output = os.path.join(output_dir, f"{base_name}_transcription.json")
                self.export_to_json(segments_ru, json_output)
                output_files["json"] = json_output
                logger.info(f"Транскрипция экспортирована в JSON: {json_output}")
                
            return output_files
            
        except Exception as e:
            logger.error(f"Ошибка при генерации субтитров: {e}")
            raise
        finally:
            # Очистка временных файлов
            self.cleanup()

    def export_to_json(self, segments, output_path):
        """
        Экспортирует сегменты в JSON формат.
        
        Args:
            segments: Список сегментов
            output_path: Путь для сохранения JSON файла
        """
        data = []
        for segment in segments:
            data.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "words": [{"word": w.word, "start": w.start, "end": w.end, "probability": w.probability} 
                         for w in segment.words] if hasattr(segment, 'words') else []
            })
            
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def cleanup(self):
        """Очищает временные файлы и директории."""
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logger.debug("Временные файлы очищены")
        except Exception as e:
            logger.warning(f"Не удалось очистить временные файлы: {e}")


def get_version_info():
    """
    Возвращает информацию о версии программы.
    
    Returns:
        dict: Словарь с информацией о версии
    """
    return VERSION_INFO

def get_version_string():
    """
    Возвращает строку с версией программы.
    
    Returns:
        str: Строка с версией в формате "X.Y.Z (release)"
    """
    return f"{__version__} ({VERSION_INFO['release']})"

def main():
    """Основная функция программы."""
    parser = argparse.ArgumentParser(
        description=f"Генератор субтитров из видео с поддержкой перевода v{__version__}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input", help="Путь к видеофайлу", nargs="?")
    parser.add_argument("--model", default="large-v3",
                      choices=["tiny", "base", "small", "medium", "large-v3"],
                      help="Модель для транскрипции")
    parser.add_argument("--max-duration", type=float, default=None,
                      help="Максимальная длительность обработки в секундах")
    parser.add_argument("--output-dir", default=None,
                      help="Директория для сохранения результатов")
    parser.add_argument("--languages", default="ru,en",
                      help="Языки для генерации субтитров (через запятую)")
    parser.add_argument("--export-json", action="store_true",
                      help="Экспортировать транскрипцию в JSON формате")
    parser.add_argument("--debug", action="store_true",
                      help="Включить отладочные сообщения")
    parser.add_argument("--version", action="version", 
                      version=f"Subtitles Generator версия {get_version_string()}\nДата сборки: {VERSION_INFO['build_date']}",
                      help="Показать версию программы и выйти")
    args = parser.parse_args()

    # Проверка наличия обязательного аргумента input
    if args.input is None:
        parser.print_help()
        return 0

    # Настройка уровня логирования
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    start_time = time.time()
    try:
        logger.info("Инициализация генератора субтитров...")
        
        # Создание генератора субтитров
        generator = SubtitlesGenerator(model_name=args.model)
        
        # Генерация субтитров
        languages = args.languages.split(',')
        output_files = generator.generate_subtitles(
            args.input, 
            output_dir=args.output_dir,
            max_duration=args.max_duration,
            languages=languages,
            export_json=args.export_json
        )
        
        # Вывод информации о результатах
        logger.info("\nГенерация субтитров завершена успешно!")
        logger.info(f"Созданные файлы:")
        for lang, path in output_files.items():
            logger.info(f"- {lang}: {path}")
            
        # Вывод статистики
        elapsed_time = time.time() - start_time
        logger.info(f"\nОбщее время выполнения: {elapsed_time:.2f} сек")

    except Exception as e:
        logger.error(f"Фатальная ошибка: {e}")
        logger.info("\nРекомендации по устранению ошибок:")
        logger.info("1. Проверьте наличие свободной памяти (не менее 5GB)")
        logger.info("2. Для длинных файлов используйте более легкую модель (например, small)")
        logger.info("3. Убедитесь в наличии интернет-соединения для загрузки моделей")
        logger.info("4. Проверьте права доступа к директориям")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
