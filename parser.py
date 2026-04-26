import requests
import pandas as pd
from tqdm import tqdm
import time


def fetch_all_items_except_2024():
    base_url = "https://api.open5e.com/v1/magicitems/"
    params = {'limit': 100}
    all_items = []

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        total_in_api = data.get('count', 0)

        print(f"Всего предметов в API (до фильтрации): {total_in_api}")
        pbar = tqdm(total=total_in_api, desc="Сбор базы")

        next_url = response.url

        while next_url:
            res = requests.get(next_url)
            res.raise_for_status()
            page_data = res.json()

            for item in page_data.get('results', []):
                # Забираем всё подряд, отфильтруем потом
                all_items.append({
                    'name': item.get('name', ''),
                    'type': item.get('type', ''),
                    'rarity': item.get('rarity', ''),
                    'requires_attunement': item.get('requires_attunement', ''),
                    'description': str(item.get('desc', '')),
                    'document_slug': str(item.get('document__slug', '')),
                    'source': str(item.get('document__title', ''))
                })
                pbar.update(1)

            next_url = page_data.get('next')
            time.sleep(0.05)

        pbar.close()

    except Exception as e:
        print(f"Ошибка при сборе: {e}")

    return pd.DataFrame(all_items)


if __name__ == "__main__":
    print("Начинаем процесс...")
    df = fetch_all_items_except_2024()

    if not df.empty:
        initial_count = len(df)

        # 1. Жесткая фильтрация 2024 года (по слагу и по названию источника)
        df = df[~df['document_slug'].str.contains('2024', na=False, case=False)]
        df = df[~df['source'].str.contains('2024', na=False, case=False)]

        # 2. Очистка от дубликатов (если имя и описание совпадают — это дубль)
        df = df.drop_duplicates(subset=['name', 'description'], keep='first')

        # 3. Причесываем текст (убираем переносы строк для Excel и нейросети)
        for col in ['description', 'name']:
            df[col] = df[col].str.replace(r'\r+|\n+', ' ', regex=True).str.strip()

        # 4. Собираем итоговую строку текста для анализа
        df['full_text'] = df['name'] + " (" + df['type'] + "): " + df['description']

        # Сохраняем в файл
        output_file = "dnd_items_final_clean.csv"
        df.to_csv(output_file, index=False, encoding='utf-8-sig', sep=';')

        # ВЫВОД СТАТИСТИКИ
        print(f"\n--- ИТОГОВЫЙ ОТЧЕТ ---")
        print(f"Было скачано (до фильтров и дублей): {initial_count}")
        print(f"Осталось чистых уникальных предметов: {len(df)}")
        print(f"Удалено (2024 год + дубликаты): {initial_count - len(df)}")

        print(f"\nРеальные уникальные источники (Slugs):")
        unique_slugs = df['document_slug'].unique()
        for slug in sorted(unique_slugs):
            count = len(df[df['document_slug'] == slug])
            print(f"- {slug}: {count} предметов")
    else:
        print("Датафрейм пуст. Что-то пошло не так при скачивании.")
    