# 🎬 Subtitles Generator

Автоматический генератор субтитров для видео с поддержкой перевода на различные языки, использующий модель Whisper от OpenAI.

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![Version](https://img.shields.io/badge/version-1.0.3-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

## ✨ Возможности

- 🔊 Автоматическая транскрипция аудио из видеофайлов
- 🌐 Перевод субтитров на разные языки (по умолчанию русский и английский)
- 📝 Интеллектуальное разделение длинных строк субтитров
- ⚡ Оптимизированная производительность на различных устройствах (CPU, CUDA, MPS)
- 📊 Экспорт результатов в форматы SRT и JSON
- 🧠 Использование современных моделей Whisper для высокого качества распознавания

## 📋 Требования

- Python 3.9 или выше
- FFmpeg (для обработки видео)
- Минимум 8 ГБ свободной оперативной памяти для моделей large-v3 (можно использовать модели меньшего размера)

## 🚀 Установка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/yourusername/subtitles_project.git
cd subtitles_project
```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

3. Установите FFmpeg (если еще не установлен):
   - **macOS**: `brew install ffmpeg`
   - **Ubuntu/Debian**: `sudo apt install ffmpeg`
   - **Windows**: Скачайте с [официального сайта](https://ffmpeg.org/download.html) или используйте [Chocolatey](https://chocolatey.org/): `choco install ffmpeg`

## 🎯 Использование

### Базовое использование

```bash
python subtitles.py /путь/к/видео.mp4
```

Это создаст файлы субтитров в формате SRT в той же директории, что и исходное видео:
- `видео_subs.srt` - субтитры на русском языке
- `видео_subs_en.srt` - субтитры на английском языке

### Расширенные опции

```bash
python subtitles.py /путь/к/видео.mp4 --model medium --languages ru,en,fr --export-json --output-dir /путь/для/сохранения
```

#### Параметры командной строки

| Параметр | Описание | Значение по умолчанию |
|----------|----------|------------------------|
| `input` | Путь к видеофайлу | (обязательный) |
| `--model` | Модель Whisper для транскрипции | `large-v3` |
| `--max-duration` | Максимальная длительность обработки в секундах | Вся длительность видео |
| `--output-dir` | Директория для сохранения результатов | Директория исходного видео |
| `--languages` | Языки для генерации субтитров (через запятую) | `ru,en` |
| `--export-json` | Экспортировать транскрипцию в JSON формате | `False` |
| `--debug` | Включить отладочные сообщения | `False` |

### Доступные модели

- `tiny`: Самая быстрая, но наименее точная модель (~39 МБ)
- `base`: Базовая модель с балансом скорости и точности (~74 МБ)
- `small`: Улучшенная точность при умеренной скорости (~244 МБ)
- `medium`: Высокая точность, но более медленная обработка (~769 МБ)
- `large-v3`: Наивысшая точность, требует больше ресурсов (~2.9 ГБ)

## 📊 Производительность

Производительность зависит от выбранной модели и доступного оборудования:

| Модель | CPU | GPU (CUDA) | Apple Silicon (MPS) |
|--------|-----|------------|---------------------|
| tiny   | ~2x | ~10x       | ~6x                 |
| base   | ~1x | ~8x        | ~5x                 |
| small  | ~0.5x | ~6x      | ~3x                 |
| medium | ~0.25x | ~4x     | ~2x                 |
| large-v3 | ~0.1x | ~2x    | ~1x                 |

*Значения указаны относительно реального времени видео (1.0x = обработка в реальном времени)

## 🔧 Устранение неполадок

### Распространенные проблемы

1. **Ошибка "No such file or directory"**:
   - Убедитесь, что путь к видеофайлу указан правильно
   - Проверьте наличие прав доступа к файлу

2. **Ошибка "CUDA out of memory"**:
   - Используйте модель меньшего размера (`small` или `base`)
   - Ограничьте длительность обработки с помощью `--max-duration`

3. **Ошибка при установке зависимостей**:
   - Убедитесь, что у вас установлена совместимая версия Python (3.9+)
   - Попробуйте создать виртуальное окружение: `python -m venv venv && source venv/bin/activate`

4. **FFmpeg не найден**:
   - Убедитесь, что FFmpeg установлен и доступен в PATH
   - На Windows: перезагрузите систему после установки FFmpeg

## 📝 История версий

### v1.0.3 (2025-03-18)
- Исправлено: Откат изменений для использования Whisper для перевода субтитров, чтобы обеспечить корректную работу и перевод на английский язык.

### v1.0.2 (2025-03-18)
- Изменено: Улучшена логика перевода субтитров для использования транскрибированных сегментов, что ускоряет процесс перевода.

### v1.0.1 (2025-03-18)
- Исправлено: Заменен moviepy на прямые вызовы ffmpeg для улучшения совместимости.
- Исправлено: Исправлена поддержка устройств MPS (Metal Performance Shaders) на Mac с Apple Silicon.
- Исправлено: Исправлен аргумент --version для корректного отображения версии программы.
- Исправлено: Аудио теперь сохраняется в текущей директории.

### v1.0.0 (2025-03-18)
- Первый стабильный релиз.
- Полная поддержка транскрипции аудио из видеофайлов
- Перевод субтитров на различные языки
- Интеллектуальное разделение длинных строк субтитров
- Оптимизированная производительность на различных устройствах (CPU, CUDA)
- Экспорт результатов в форматы SRT и JSON
- Использование моделей Whisper для высокого качества распознавания

## 📝 Лицензия

Этот проект распространяется под лицензией MIT. Подробности в файле [LICENSE](LICENSE).

## 📋 Версионирование

Проект следует принципам [семантического версионирования](https://semver.org/lang/ru/).

Формат версии: `МАЖОРНАЯ.МИНОРНАЯ.ПАТЧ`

- **МАЖОРНАЯ** версия - несовместимые изменения API
- **МИНОРНАЯ** версия - добавление функциональности с обратной совместимостью
- **ПАТЧ** версия - исправления ошибок с обратной совместимостью

## 🙏 Благодарности

- [OpenAI Whisper](https://github.com/openai/whisper) - за разработку модели Whisper
- [Faster Whisper](https://github.com/guillaumekln/faster-whisper) - за оптимизированную реализацию Whisper
- [MoviePy](https://zulko.github.io/moviepy/) - за инструменты обработки видео
- [SRT](https://github.com/cdown/srt) - за работу с форматом субтитров
