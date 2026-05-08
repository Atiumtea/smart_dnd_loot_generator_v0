import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
from tqdm import tqdm
import time

def create_resilient_session():
    """Создает сессию, которая не падает при обрывах связи"""
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[ 502, 503, 504, 429 ])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

def fetch_all_items_except_2024():
    base_url = "https://api.open5e.com/v1/magicitems/"
    params = {'limit': 100}
    all_items = []
    session = create_resilient_session()

    try:
        response = session.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        total_in_api = data.get('count', 0)

        print(f"Всего предметов в API (до фильтрации): {total_in_api}")
        pbar = tqdm(total=total_in_api, desc="Сбор базы")

        next_url = response.url

        while next_url:
            res = session.get(next_url, timeout=10)
            res.raise_for_status()
            page_data = res.json()

            for item in page_data.get('results', []):
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

        # ИСХОДНИК ИСПРАВЛЕН: Точечное удаление вместо ковровой бомбардировки
        # Удаляем только конкретные слаги 2024 года, чтобы не задеть нормальные предметы
        banned_slugs = ['dnd-2024-core', 'free-rules-2024']
        df = df[~df['document_slug'].isin(banned_slugs)]

        # Очистка от дубликатов
        df = df.drop_duplicates(subset=['name', 'description'], keep='first')

        for col in ['description', 'name']:
            df[col] = df[col].str.replace(r'\r+|\n+', ' ', regex=True).str.strip()

        # ИСХОДНИК ИСПРАВЛЕН: Структурированный промпт для нейросети
        # SentenceTransformers гораздо лучше понимает структуру "Ключ: Значение"
        df['full_text'] = "Item Name: " + df['name'] + ". Type: " + df['type'] + ". Description: " + df['description']

        output_file = "dnd_items_final_clean.csv"
        df.to_csv(output_file, index=False, encoding='utf-8-sig', sep=';')

        print(f"\n--- ИТОГОВЫЙ ОТЧЕТ ---")
        print(f"Было скачано: {initial_count}")
        print(f"Осталось: {len(df)}")
        print(f"Удалено: {initial_count - len(df)}")
    else:
        print("Датафрейм пуст. Что-то пошло не так при скачивании.")