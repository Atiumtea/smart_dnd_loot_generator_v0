import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle


def create_knowledge_base():
    print("1. Загрузка очищенных данных...")
    try:
        # Указываем правильный разделитель!
        df = pd.read_csv('dnd_items_final_clean.csv', sep=';')
    except FileNotFoundError:
        print("Ошибка: Файл 'dnd_items_final_clean.csv' не найден. Убедись, что парсер отработал успешно.")
        return

    # Проверка на пустые значения (иногда парсер может оставить NaN, что сломает нейросеть)
    df['full_text'] = df['full_text'].fillna("")
    print(f"Успешно загружено {len(df)} предметов.")

    print("\n2. Загрузка языковой модели (SentenceTransformer)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("\n3. Векторизация текстов (превращаем слова в числа)...")
    # Превращаем колонку в список строк
    texts = df['full_text'].tolist()

    # encode() сам батчит данные. show_progress_bar покажет ползунок загрузки.
    # convert_to_tensor=False оставляет векторы в виде numpy массивов, чтобы их было легко сохранить.
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_tensor=False)

    # Записываем векторы обратно в датафрейм в новую колонку 'embedding'
    df['embedding'] = list(embeddings)

    print("\n4. Сохранение базы знаний в .pkl...")
    # Сохраняем весь датафрейм (теперь он содержит и тексты, и их векторы)
    with open('dnd_knowledge_base.pkl', 'wb') as f:
        pickle.dump(df, f)

    print("Готово! Файл 'dnd_knowledge_base.pkl' успешно создан и готов к работе в боевом генераторе.")


if __name__ == "__main__":
    create_knowledge_base()