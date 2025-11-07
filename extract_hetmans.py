import pdfplumber
import re
import os
from pathlib import Path

# Шляхи
pdf_path = "Usi_hetmany_Ukrainy.pdf"
output_dir = "data/hetman_files/"
os.makedirs(output_dir, exist_ok=True)

# Список гетьманів у правильному порядку
hetmans = [
    "Дмитро Байда-Вишневецький",
    "Петро Конашевич-Сагайдачний",
    "Богдан-Зиновій Михайлович Хмельницький",
    "Іван Остапович Виговський",
    "Юрій-Гедеон Венжик Богданович Хмельницький(син)",
    "Павло Іванович Моржковський-Тетеря",
    "Іван Мартинович Брюховецький",
    "Петро Дорофійович Дорошенко",
    "Михайло Степанович Ханенко",
    "Дем’ян Гнатович Многогрішний",
    "Іван Самійлович Самойлович",
    "Іван Степанович Мазепа",
    "Пилип Степанович Орлик",
    "Іван Ілліч Скоропадський",
    "Данило Павлович Апостол",
    "Кирило Григорович Розумовський",
    "Павло Петрович Скоропадський"
]

def is_quote_start(text):
    """Перевіряє, чи сторінка починається з цитати («...»)"""
    if not text:
        return False
    match = re.match(r'\s*«[^»]+»', text.strip())
    return bool(match)

def pdf_to_sections(pdf_path, output_dir, start_page=7, end_page=412):
    """Розбиття PDF на розділи за новою сторінкою + цитатою на початку, зберігаючи весь текст"""
    current_text = []
    section_count = 1
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                if start_page <= page_num <= end_page:
                    print(f"Обробка сторінки {page_num}...")
                    text = page.extract_text()
                    if text and len(text.strip()) > 10:
                        if is_quote_start(text):
                            if current_text:
                                output_file = os.path.join(output_dir, f"{section_count}.txt")
                                with open(output_file, 'w', encoding='utf-8') as f:
                                    f.write('\n'.join(current_text).strip())
                                print(f"Збережено розділ {section_count}: {output_file} (довжина: {len(''.join(current_text))} знаків)")
                                section_count += 1
                            current_text = [text]  # Зберігаємо цитату як початок нового розділу
                        else:
                            current_text.append(text)  # Додаємо весь текст, включаючи попередній контекст
                    else:
                        print(f"Сторінка {page_num} порожня.")
            if current_text:
                output_file = os.path.join(output_dir, f"{section_count}.txt")
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(current_text).strip())
                print(f"Збережено останній розділ {section_count}: {output_file} (довжина: {len(''.join(current_text))} знаків)")

    except Exception as e:
        print(f"Помилка: {e}")

def merge_extra_file(output_dir):
    """Приєднує зайвий файл 13.txt до 12.txt"""
    files = sorted([f for f in os.listdir(output_dir) if f.endswith('.txt')], key=lambda x: int(x.split('.')[0]))
    if len(files) == 18:  # Очікуємо 18 файлів через помилку на 289
        extra_file = files[12]  # 13.txt (індекс 12, бо нумерація з 0)
        prev_file = files[11]  # 12.txt (індекс 11)
        extra_path = os.path.join(output_dir, extra_file)
        prev_path = os.path.join(output_dir, prev_file)

        with open(extra_path, 'r', encoding='utf-8') as f:
            extra_content = f.read()
        with open(prev_path, 'a', encoding='utf-8') as f:
            f.write('\n\n' + extra_content.strip())
        os.remove(extra_path)
        print(f"Приєднано {extra_file} до {prev_file} і видалено.")

def process_hetman_files(output_dir, hetmans):
    """Обробка файлів: нумерування, видалення цитат на початку, прибирання номерів сторінок, заміна ПІБ, перейменування"""
    # Отримуємо список файлів і сортуємо за номером
    files = sorted([f for f in os.listdir(output_dir) if f.endswith('.txt')], key=lambda x: int(x.split('.')[0]))

    # 1. Нова нумерація (1.txt до 17.txt)
    for i, filename in enumerate(files, 1):
        old_path = os.path.join(output_dir, filename)
        new_filename = os.path.join(output_dir, f"{i}.txt")
        os.rename(old_path, new_filename)
        print(f"Перейменовано {filename} на {i}.txt")

    # Оновлюємо список файлів після перейменування
    files = sorted([f for f in os.listdir(output_dir) if f.endswith('.txt')], key=lambda x: int(x.split('.')[0]))

    # 2-5. Обробка вмісту та перейменування
    for i, filename in enumerate(files, 1):
        file_path = os.path.join(output_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 2. Видаляємо цитату ТІЛЬКИ на початку файлу (новий регулярний вираз)
        if re.match(r'^\s*«[\s\S]+?»', content):
            content = re.sub(
                r'^(\s*«[\s\S]+?»\s*)([\s\S]+?\([\s\S]+?\)\s*)',
                r'\2',
                content,
                count=1,
                flags=re.DOTALL | re.MULTILINE
            )

        # 3. Прибираємо номери сторінок (числа на кінці рядків) і наступні пусті рядки
        content = re.sub(r'\b\d+\b\s*(?=\s*$)', '', content, flags=re.MULTILINE)
        content = re.sub(r'^\s*\n', '', content, flags=re.MULTILINE)  # Видаляємо порожні рядки

        # 4. Замінюємо коряве ПІБ на нормальне
        if i <= len(hetmans):
            correct_name = hetmans[i - 1]
            content = re.sub(r'^[А-ЯІЇЄҐ][а-яіїєґ\s\-\w]*(?=\n)', correct_name, content, flags=re.MULTILINE, count=1)

        # Зберігаємо оновлений вміст
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        # 5. Перейменовуємо файл на прізвище
        surname = hetmans[i - 1].split()[-1]
        new_filename = os.path.join(output_dir, f"{surname}.txt")
        if os.path.exists(new_filename):
            os.remove(new_filename)  # Видаляємо існуючий файл перед перейменуванням
        os.rename(file_path, new_filename)
        print(f"Перейменовано {filename} на {surname}.txt")

    print(f"Обробка завершена. Створено 17 файлів у {output_dir}.")

if __name__ == "__main__":
    if not os.path.exists(pdf_path):
        print(f"Файл {pdf_path} не знайдено!")
    else:
        pdf_to_sections(pdf_path, output_dir, start_page=7, end_page=412)
        merge_extra_file(output_dir)
        process_hetman_files(output_dir, hetmans)
        print(f"Повний процес завершено. Створено 17 оброблених файлів у {output_dir}.")