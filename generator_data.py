import pandas as pd
import numpy as np
import random
import math
from models import ITEM_TYPES, get_type_ohe


def smooth_normalize(score, midpoint=0.25, steepness=15):
    return 1 / (1 + math.exp(-steepness * (score - midpoint)))


def calculate_target_y(location_score, party_score, story_importance, level_rarity_delta, is_duplicate, synergy_flag):
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

    # 5. Редкость (Delta) - ИСПРАВЛЕННАЯ НЕЛИНЕЙНАЯ ЛОГИКА
    if level_rarity_delta > 0:
        if level_rarity_delta == 1:
            # +1 тир: Босс (imp=1) прощает это, рядовой бой (imp=0) жестко штрафует
            penalty = 1.0 - (0.5 * (1.0 - story_importance))
            y *= penalty
        elif level_rarity_delta == 2:
            # +2 тира: Перебор для любого этапа. Жестко режем шансы.
            y *= 0.2 + (0.1 * story_importance)
        else:
            # +3 и +4 тира (Легендарка на 1-4 лвл): Математическое уничтожение таргета.
            y *= 0.01

    elif level_rarity_delta < 0:
        # Штраф за мусор (слишком слабая вещь)
        # Если это Босс (imp=1), мусор падать не должен -> жесткий штраф
        penalty = 1.0 - (abs(level_rarity_delta) * 0.15 * story_importance)
        y *= max(0.1, penalty)

    # 6. Дубликаты
    if is_duplicate == 1.0:
        y *= 0.1

    y += random.gauss(0, 0.02)  # Легкий шум
    return float(round(np.clip(y, 0.0, 1.0), 4))


def generate_dnd_dataset(num_samples=25000):
    data = []

    for _ in range(num_samples):
        if random.random() < 0.7:
            loc_s = round(np.clip(random.gauss(0.3, 0.15), 0.0, 1.0), 4)
            par_s = round(np.clip(random.gauss(0.3, 0.15), 0.0, 1.0), 4)
        else:
            loc_s = round(random.uniform(0.0, 1.0), 4)
            par_s = round(random.uniform(0.0, 1.0), 4)
        imp = round(random.uniform(0.0, 1.0), 4)
        delta = random.randint(-4, 4)
        is_dup = 1.0 if random.random() < 0.05 else 0.0

        chosen_type = random.choice(ITEM_TYPES)
        syn = 1.0 if random.random() < 0.8 else 0.0

        target = calculate_target_y(loc_s, par_s, imp, delta, is_dup, syn)
        type_ohe = get_type_ohe(chosen_type)

        row = {
            'loc_score': loc_s, 'party_score': par_s, 'story_importance': imp,
            'level_rarity_delta': delta, 'is_duplicate': is_dup,
            'synergy_flag': syn, 'target_y': target
        }

        for i, t in enumerate(ITEM_TYPES):
            row[f'type_{t.replace(" ", "_")}'] = type_ohe[i]

        data.append(row)

    return pd.DataFrame(data)

if __name__ == "__main__":
    print("🚀 Генерация умного датасета (Сложные нелинейные штрафы)...")
    df = generate_dnd_dataset()
    df.to_csv("dnd_mlp_training_data.csv", index=False, sep=';')
    print("✅ Готово!")
    print(f"Средний таргет Y: {df['target_y'].mean():.4f}")