import pandas as pd
import numpy as np
import random
import math
from models import ITEM_TYPES, get_type_ohe

def smooth_normalize(score, midpoint=0.25, steepness=15):
    """
    Плавная нормализация через Сигмоиду.
    midpoint=0.25 - центр кривой (среднее значение наших векторов).
    steepness=15 - крутизна S-кривой (настроена так, чтобы 0.45 давало ~0.95).
    """
    return 1 / (1 + math.exp(-steepness * (score - midpoint)))


def calculate_target_y(location_score, party_score, story_importance, level_rarity_delta, is_duplicate,
                       synergy_flag):
    # 1. Плавная нелинейная нормализация
    norm_loc = smooth_normalize(location_score)
    norm_party = smooth_normalize(party_score)

    # 2. Базовая семантика
    semantic_base = max(norm_loc, norm_party) * 0.7 + min(norm_loc, norm_party) * 0.3

    # 3. Множитель важности
    y = semantic_base * (0.4 + 0.6 * story_importance)

    # 4. Строгая синергия
    if synergy_flag == 0:
        y *= 0.15
    else:
        y *= 1.1

    # 5. Редкость (Delta)
    if level_rarity_delta > 0:
        penalty = 1.0 - (level_rarity_delta * 0.4 * (1.0 - story_importance))
        y *= max(0.1, penalty)
    elif level_rarity_delta < -1:
        y *= max(0.2, 1.0 - (abs(level_rarity_delta) * 0.2 * story_importance))

    # 6. Дубликаты
    if is_duplicate == 1:
        y *= 0.1

    y += random.gauss(0, 0.03)  # Легкий шум
    return float(round(np.clip(y, 0.0, 1.0), 4))


def generate_dnd_dataset(num_samples=25000):
    data = []

    for _ in range(num_samples):
        loc_s = round(np.clip(random.gauss(0.25, 0.08), 0.0, 1.0), 4)
        par_s = round(np.clip(random.gauss(0.25, 0.08), 0.0, 1.0), 4)
        imp = round(random.uniform(0.0, 1.0), 4)
        delta = random.randint(-4, 4)
        is_dup = 1.0 if random.random() < 0.05 else 0.0

        # Выбираем случайный ТИП строкой, а не числом
        chosen_type = random.choice(ITEM_TYPES)
        syn = 1.0 if random.random() < 0.8 else 0.0

        target = calculate_target_y(loc_s, par_s, imp, delta, is_dup, chosen_type, syn)

        # Получаем One-Hot массив (9 колонок)
        type_ohe = get_type_ohe(chosen_type)

        # Собираем базовые фичи
        row = {
            'loc_score': loc_s, 'party_score': par_s, 'story_importance': imp,
            'level_rarity_delta': delta, 'is_duplicate': is_dup,
            'synergy_flag': syn, 'target_y': target
        }

        # Динамически добавляем 9 колонок с типами (type_weapon, type_armor и т.д.)
        for i, t in enumerate(ITEM_TYPES):
            row[f'type_{t.replace(" ", "_")}'] = type_ohe[i]

        data.append(row)

    return pd.DataFrame(data)


if __name__ == "__main__":
    print("🚀 Генерация умного датасета (Сигмоида + Гаусс)...")
    df = generate_dnd_dataset()
    df.to_csv("dnd_mlp_training_data.csv", index=False, sep=';')
    print("✅ Готово!")
    print(f"Средний таргет Y: {df['target_y'].mean():.4f}")