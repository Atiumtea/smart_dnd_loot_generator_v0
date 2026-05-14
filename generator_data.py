import pandas as pd
import numpy as np
import random
import math
from models import ITEM_TYPES, get_type_ohe


def smooth_normalize(score, midpoint=0.25, steepness=15):
    return 1 / (1 + math.exp(-steepness * (score - midpoint)))


def calculate_target_y(location_score, party_score, story_importance, level_rarity_delta, is_duplicate, synergy_flag,
                       item_type_str):
    norm_loc = smooth_normalize(location_score)
    norm_party = smooth_normalize(party_score)

    # 1.
    if norm_loc < 0.15:
        semantic_base = min(norm_loc, norm_party)  # Берем худшее
    else:
        semantic_base = max(norm_loc, norm_party) * 0.7 + min(norm_loc, norm_party) * 0.3

    y = semantic_base * (0.4 + 0.6 * story_importance)

    # 2. Синергия
    if synergy_flag == 0:
        y *= 0.15
    else:
        y *= 1.1

    # 3. Нелинейная Редкость (Delta)
    if level_rarity_delta > 0:
        if level_rarity_delta == 1:
            y *= 1.0 - (0.5 * (1.0 - story_importance))
        elif level_rarity_delta == 2:
            y *= 0.15 + (0.1 * story_importance)
        else:
            y *= 0.01

    elif level_rarity_delta < 0:
        base_penalty = abs(level_rarity_delta) * 0.25 * story_importance
        # Если delta -3 или -4 на боссе, penalty будет > 0.75
        y *= max(0.05, 1.0 - base_penalty)

    # 4. Умные Дубликаты (Расходники прощаются)
    if is_duplicate == 1.0:
        consumables = ['potion', 'scroll']
        if any(c in item_type_str.lower() for c in consumables):
            y *= 0.8
        else:
            y *= 0.1

    y += random.gauss(0, 0.02)
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

        target = calculate_target_y(loc_s, par_s, imp, delta, is_dup, syn, chosen_type)
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