import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
import pickle


def create_knowledge_base():
    print("1. Загрузка очищенных данных...")
    try:
        df = pd.read_csv('dnd_items_final_clean.csv', sep=';')
    except FileNotFoundError:
        print("Ошибка: Файл 'dnd_items_final_clean.csv' не найден. Сначала запусти parser.py.")
        return

    df['full_text'] = df['full_text'].fillna("")
    print(f"Успешно загружено {len(df)} предметов.")

    print("\n2. Загрузка языковой модели (SentenceTransformer)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("\n3. Векторизация текстов...")
    texts = df['full_text'].tolist()
    # Батчинг на месте (64)
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_tensor=False)

    df['embedding'] = list(embeddings)

    print("\n4. Сохранение базы в .pkl (Легаси для скриптов обучения)...")
    with open('dnd_knowledge_base.pkl', 'wb') as f:
        pickle.dump(df, f)

    print("\n5. Создание векторной базы ChromaDB (Для боевого генератора)...")
    client = chromadb.PersistentClient(path="./dnd_vector_db")

    # Удаляем старую коллекцию, если она есть, чтобы обновить данные начисто
    try:
        client.delete_collection("magic_items")
    except:
        pass

    collection = client.create_collection(
        name="magic_items",
        metadata={"hnsw:space": "cosine"}  # Используем косинусное расстояние
    )

    # Безопасная упаковка метаданных (ChromaDB не любит пустые значения и NaN)
    metadatas = []
    for _, row in df.iterrows():
        metadatas.append({
            'name': str(row.get('name', 'Unknown')),
            'type': str(row.get('type', 'wondrous item')),
            'rarity': str(row.get('rarity', 'common'))
        })

    ids = [str(i) for i in range(len(df))]

    # Добавляем данные в базу (пачками по 5000 для стабильности)
    batch_size = 5000
    for i in range(0, len(texts), batch_size):
        collection.add(
            embeddings=embeddings[i:i + batch_size].tolist(),
            documents=texts[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
            ids=ids[i:i + batch_size]
        )

    print("✅ Готово! Векторная база ChromaDB успешно создана (папка 'dnd_vector_db').")


if __name__ == "__main__":
    create_knowledge_base()