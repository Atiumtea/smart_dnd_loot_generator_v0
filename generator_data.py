import pandas as pd
import numpy as np
import random


def calculate_target_y(location_score, party_score, story_importance, level_rarity_delta, is_duplicate, type_id,
                       synergy_flag):
    # НОРМАЛИЗАЦИЯ: Растягиваем реальные скоры (0.1 - 0.4) в шкалу (0.25 - 1.0)
    norm_loc = min(1.0, location_score * 2.5)
    norm_party = min(1.0, party_score * 2.5)

    # 1. Базовая семантика (теперь использует нормализованные значения)
    semantic_base = max(norm_loc, norm_party) * 0.7 + min(norm_loc, norm_party) * 0.3

    y = semantic_base * (0.4 + 0.6 * story_importance)

    if synergy_flag == 0:
        y *= 0.15
    else:
        y *= 1.1

    if level_rarity_delta > 0:
        penalty = 1.0 - (level_rarity_delta * 0.4 * (1.0 - story_importance))
        y *= max(0.1, penalty)
    elif level_rarity_delta < -1:
        y *= max(0.2, 1.0 - (abs(level_rarity_delta) * 0.2 * story_importance))

    if is_duplicate == 1: y *= 0.1
    y += random.gauss(0, 0.03)
    return float(round(np.clip(y, 0.0, 1.0), 4))


def generate_dnd_dataset(num_samples=25000):
    data = []
    types = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for _ in range(num_samples):
        # ГЕНЕРИРУЕМ РЕАЛИСТИЧНЫЕ СКОРЫ!
        loc_s = round(random.uniform(0.05, 0.45), 4)
        par_s = round(random.uniform(0.05, 0.45), 4)
        imp = round(random.uniform(0.0, 1.0), 4)
        delta = random.randint(-4, 4)
        is_dup = 1 if random.random() < 0.05 else 0
        t_id = random.choice(types)
        # 80% предметов подходят партии, 20% - нет (для обучения фильтрации)
        syn = 1.0 if random.random() < 0.8 else 0.0

        target = calculate_target_y(loc_s, par_s, imp, delta, is_dup, t_id, syn)

        data.append({
            'loc_score': loc_s, 'party_score': par_s, 'story_importance': imp,
            'level_rarity_delta': delta, 'is_duplicate': is_dup,
            'type_id': t_id, 'synergy_flag': syn, 'target_y': target
        })
    return pd.DataFrame(data)


if __name__ == "__main__":
    df = generate_dnd_dataset()
    df.to_csv("dnd_mlp_training_data.csv", index=False, sep=';')
    print("✅ Синтетика на 7 признаков готова!")