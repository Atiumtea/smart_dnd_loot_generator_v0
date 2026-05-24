import os
import time
import pickle
import requests
import pandas as pd
import chromadb
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sentence_transformers import SentenceTransformer

def create_resilient_session():
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504, 429])
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

def build_knowledge_base():
    print("\n[ЭТАП 1] Загрузка и очистка данных...")
    df = fetch_all_items_except_2024()

    if df.empty:
        print("Ошибка: Датафрейм пуст. Что-то пошло не так при скачивании.")
        return

    initial_count = len(df)

    banned_slugs = ['dnd-2024-core', 'free-rules-2024']
    df = df[~df['document_slug'].isin(banned_slugs)]

    df = df.drop_duplicates(subset=['name', 'description'], keep='first')

    for col in ['description', 'name']:
        df[col] = df[col].str.replace(r'\r+|\n+', ' ', regex=True).str.strip()

    df['full_text'] = "Item Name: " + df['name'] + ". Type: " + df['type'] + ". Description: " + df['description']
    df['full_text'] = df['full_text'].fillna("")

    df.to_csv("dnd_items_final_clean.csv", index=False, encoding='utf-8-sig', sep=';')

    print(f"Отчет: скачано {initial_count}, удалено {initial_count - len(df)}, итого предметов: {len(df)}")

    print("\n[ЭТАП 2] Загрузка языковой модели (SentenceTransformer)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("\n[ЭТАП 3] Векторизация текстов...")
    texts = df['full_text'].tolist()
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_tensor=False)
    df['embedding'] = list(embeddings)

    print("\n[ЭТАП 4] Сохранение базы в .pkl (Легаси для скриптов обучения)...")
    with open('dnd_knowledge_base.pkl', 'wb') as f:
        pickle.dump(df, f)

    print("\n[ЭТАП 5] Создание векторной базы ChromaDB...")
    client = chromadb.PersistentClient(path="./dnd_vector_db")

    try:
        client.delete_collection("magic_items")
    except:
        pass

    collection = client.create_collection(
        name="magic_items",
        metadata={"hnsw:space": "cosine"}
    )

    metadatas = []
    for _, row in df.iterrows():
        metadatas.append({
            'name': str(row.get('name', 'Unknown')),
            'type': str(row.get('type', 'wondrous item')),
            'rarity': str(row.get('rarity', 'common')),
            'source': str(row.get('source', 'Unknown'))
        })

    ids = [str(i) for i in range(len(df))]

    batch_size = 5000
    for i in range(0, len(texts), batch_size):
        collection.add(
            embeddings=embeddings[i:i + batch_size].tolist(),
            documents=texts[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
            ids=ids[i:i + batch_size]
        )

    print("\n✅ Готово! Единый пайплайн данных успешно выполнен.")

if __name__ == "__main__":
    build_knowledge_base()