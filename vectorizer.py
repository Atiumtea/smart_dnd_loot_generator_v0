import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle


def create_knowledge_base():
    print("1. Загрузка очищенных данных...")
    try:
        df = pd.read_csv('dnd_items_final_clean.csv', sep=';')
    except FileNotFoundError:
        print("Ошибка: Файл 'dnd_items_final_clean.csv' не найден. Убедись, что парсер отработал успешно.")
        return

    df['full_text'] = df['full_text'].fillna("")
    print(f"Успешно загружено {len(df)} предметов.")

    print("\n2. Загрузка языковой модели (SentenceTransformer)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("\n3. Векторизация текстов (превращаем слова в числа)...")
    texts = df['full_text'].tolist()

    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_tensor=False)

    df['embedding'] = list(embeddings)

    print("\n4. Сохранение базы знаний в .pkl...")
    with open('dnd_knowledge_base.pkl', 'wb') as f:
        pickle.dump(df, f)

    print("Готово! Файл 'dnd_knowledge_base.pkl' успешно создан и готов к работе в боевом генераторе.")


if __name__ == "__main__":
    create_knowledge_base()