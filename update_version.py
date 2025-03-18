#!/usr/bin/env python3
"""
Скрипт для обновления версии проекта
"""

import re
import os
import datetime
import argparse
import json

# Путь к основному файлу проекта
MAIN_FILE = "subtitles.py"
# Путь к файлу README.md
README_FILE = "README.md"
# Путь к файлу CHANGELOG.md
CHANGELOG_FILE = "CHANGELOG.md"

def update_version_in_file(file_path, current_version, new_version):
    """Обновляет версию в файле"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Обновление версии в файле
    content = content.replace(f'__version__ = "{current_version}"', f'__version__ = "{new_version}"')
    
    # Обновление информации о версии
    version_parts = new_version.split('.')
    if len(version_parts) >= 3:
        major, minor, patch = version_parts[0], version_parts[1], version_parts[2]
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        
        # Обновление VERSION_INFO
        version_info_pattern = r'VERSION_INFO = \{[^}]+\}'
        new_version_info = f'''VERSION_INFO = {{
    "major": {major},
    "minor": {minor},
    "patch": {patch},
    "release": "stable",
    "build_date": "{today}"
}}'''
        content = re.sub(version_info_pattern, new_version_info, content)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def update_version_in_readme(file_path, current_version, new_version):
    """Обновляет версию в README.md"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Обновление бейджа версии
    content = re.sub(
        r'!\[Version\]\(https://img\.shields\.io/badge/version-[^-]+-brightgreen\)',
        f'![Version](https://img.shields.io/badge/version-{new_version}-brightgreen)',
        content
    )
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def update_changelog(file_path, new_version, changes):
    """Добавляет новую версию в CHANGELOG.md"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    new_version_entry = f'''## [{new_version}] - {today}

{changes}

'''
    
    # Вставляем новую версию после заголовка
    pattern = r'# История изменений.*?\n\n'
    replacement = f'\\g<0>{new_version_entry}'
    content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def get_current_version():
    """Получает текущую версию из файла"""
    with open(MAIN_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
    
    match = re.search(r'__version__ = "([^"]+)"', content)
    if match:
        return match.group(1)
    return "0.0.0"

def main():
    parser = argparse.ArgumentParser(description="Обновление версии проекта")
    parser.add_argument('new_version', help='Новая версия в формате X.Y.Z')
    parser.add_argument('--changes', help='Описание изменений для CHANGELOG.md', default='')
    args = parser.parse_args()
    
    current_version = get_current_version()
    print(f"Текущая версия: {current_version}")
    print(f"Новая версия: {args.new_version}")
    
    # Обновление версии в основном файле
    update_version_in_file(MAIN_FILE, current_version, args.new_version)
    print(f"Обновлена версия в {MAIN_FILE}")
    
    # Обновление версии в README.md
    update_version_in_readme(README_FILE, current_version, args.new_version)
    print(f"Обновлена версия в {README_FILE}")
    
    # Обновление CHANGELOG.md
    if args.changes:
        update_changelog(CHANGELOG_FILE, args.new_version, args.changes)
        print(f"Обновлен {CHANGELOG_FILE}")
    
    print("Версия успешно обновлена!")

if __name__ == "__main__":
    main()
