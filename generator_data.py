import pandas as pd
import numpy as np
import random


def calculate_target_y(location_score, party_score, story_importance, level_rarity_delta, is_duplicate):
    """
    Улучшенная функция-оракул с двойным семантическим входом (5 признаков).
    """
    # 1. Объединяем семантику (базовый вес) - 50/50
    semantic_base = max(location_score, party_score) * 0.7 + min(location_score, party_score) * 0.3

    # 2. Плавный множитель важности (от 0.2 до 1.0)
    importance_mult = 0.4 + (0.6 * story_importance)
    y = semantic_base * importance_mult

    # 3. Обработка предметов, которые КРУЧЕ уровня партии (delta > 0)
    if level_rarity_delta > 0:
        if level_rarity_delta == 1 and story_importance > 0.6:
            # Бонус за эпичность при переходе на 1 тир выше
            epic_bonus = 0.25 * (story_importance ** 2)
            y += epic_bonus
        else:
            # Штраф за слишком высокий уровень
            penalty = 1.0 - (level_rarity_delta * 0.4 * (1.0 - story_importance))
            y *= max(0.1, penalty)

    # 4. Обработка предметов, которые СЛАБЕЕ уровня партии (delta < 0)
    elif level_rarity_delta < -1:
        # Чем важнее бой, тем сильнее штраф за слабый лут
        weakness_penalty = 1.0 - (abs(level_rarity_delta) * 0.25 * story_importance)
        y *= max(0.2, weakness_penalty)

    # 5. Штраф за дубликаты
    if is_duplicate == 1:
        y *= 0.1

    # 6. Добавление шума
    noise = random.gauss(0, 0.05)
    y += noise

    return float(round(np.clip(y, 0.0, 1.0), 4))


def generate_dnd_dataset(num_samples=20000):
    """
    Генератор датасета с 5 входными признаками.
    """
    data = []

    for _ in range(num_samples):
        # Генерируем два независимых семантических скора
        location_score = round(random.uniform(0.4, 0.95), 4)
        party_score = round(random.uniform(0.4, 0.95), 4)

        story_importance = round(random.uniform(0.0, 1.0), 4)
        level_rarity_delta = random.randint(-4, 4)

        # Умная генерация дубликатов
        if level_rarity_delta > 0:
            is_duplicate = 0
        else:
            is_duplicate = 1 if random.random() < 0.07 else 0

        # ВЫЗОВ ФУНКЦИИ (теперь все 5 аргументов на месте)
        target_y = calculate_target_y(
            location_score,
            party_score,
            story_importance,
            level_rarity_delta,
            is_duplicate
        )

        data.append({
            'location_score': location_score,
            'party_score': party_score,
            'story_importance': story_importance,
            'level_rarity_delta': level_rarity_delta,
            'is_duplicate': is_duplicate,
            'target_y': target_y
        })

    return pd.DataFrame(data)


if __name__ == "__main__":
    print("🚀 Генерация нового датасета (5 признаков)...")
    df = generate_dnd_dataset(num_samples=25000)  # Чуть больше данных для 5 входов

    filename = "dnd_mlp_training_data.csv"
    df.to_csv(filename, index=False, sep=';')

    print(f"✅ Готово! Файл '{filename}' создан.")
    print(f"Средний таргет: {df['target_y'].mean():.4f}")