import os
import warnings
import logging

# ==========================================
# 0. ГЛУШИТЕЛЬ ЛОГОВ (Должен быть в самом верху)
# ==========================================
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
warnings.filterwarnings("ignore")
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import random
import re
from sentence_transformers import SentenceTransformer, util

# Словарь перевода типов в числа (для нейросети)
TYPE_MAP = {
    'weapon': 0.1,
    'armor': 0.2,
    'potion': 0.3,
    'ring': 0.4,
    'scroll': 0.5,
    'wand': 0.6,
    'staff': 0.7,
    'rod': 0.8,
    'wondrous item': 0.9,
}

CLASS_SYNERGY = {
    'barbarian': ['weapon', 'potion', 'ring', 'wondrous item'],
    'monk': ['weapon', 'potion', 'ring', 'wondrous item'],  # Монахам не нужна броня
    'fighter': ['weapon', 'armor', 'potion', 'ring', 'wondrous item'],
    'rogue': ['weapon', 'armor', 'potion', 'ring', 'wondrous item'],

    'paladin': ['weapon', 'armor', 'potion', 'ring', 'scroll', 'wondrous item'],
    'ranger': ['weapon', 'armor', 'potion', 'ring', 'scroll', 'wondrous item'],

    'cleric': ['weapon', 'armor', 'potion', 'ring', 'scroll', 'rod', 'staff', 'wondrous item'],
    'druid': ['weapon', 'armor', 'potion', 'ring', 'scroll', 'staff', 'wondrous item'],
    'bard': ['weapon', 'armor', 'potion', 'ring', 'scroll', 'wand', 'staff', 'wondrous item'],

    'wizard': ['potion', 'ring', 'scroll', 'wand', 'staff', 'rod', 'wondrous item'],
    'sorcerer': ['potion', 'ring', 'scroll', 'wand', 'staff', 'rod', 'wondrous item'],
    'warlock': ['weapon', 'potion', 'ring', 'scroll', 'wand', 'staff', 'rod', 'wondrous item'],

    'artificer': ['weapon', 'armor', 'potion', 'ring', 'scroll', 'wand', 'staff', 'rod', 'wondrous item']
    # Используют почти всё
}

# ==========================================
# 1. АРХИТЕКТУРА СЕТИ
# ==========================================
class DnDItemRanker(nn.Module):
    def __init__(self, input_size=7):
        super(DnDItemRanker, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


# ==========================================
# 2. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================================
def get_rarity_val(rarity_str, expected_rarity=3):
    """
    Умный парсер редкости. Подстраивает 'varies' и мульти-тир предметы
    под текущий уровень партии.
    """
    r = str(rarity_str).lower()

    # 1. Если редкость плавающая - предмет ИДЕАЛЬНО подстраивается под партию
    if 'varies' in r:
        return expected_rarity

    found_rarities = []

    # 2. Ищем все возможные редкости в строке
    if 'artifact' in r: found_rarities.append(6)
    if 'legendary' in r: found_rarities.append(5)

    if 'very rare' in r:
        found_rarities.append(4)
        r = r.replace('very rare', '')  # Вырезаем, чтобы не было ложного 'rare'

    if 'uncommon' in r:
        found_rarities.append(2)
        r = r.replace('uncommon', '')  # Вырезаем, чтобы не было ложного 'common'

    # Точный поиск слов (границы слов \b)
    if re.search(r'\brare\b', r): found_rarities.append(3)
    if re.search(r'\bcommon\b', r): found_rarities.append(1)

    # 3. Базовый случай: ничего не нашли
    if not found_rarities:
        return 1

    # 4. МАГИЯ ДИНАМИКИ: Если найдено несколько редкостей (например, масштабируемый лут)
    # Выбираем ту, которая ближе всего к ожидаемой редкости партии!
    best_rarity = min(found_rarities, key=lambda x: abs(x - expected_rarity))

    return best_rarity


def get_expected_rarity_for_level(level):
    if level <= 4:
        return 2
    elif level <= 10:
        return 3
    elif level <= 16:
        return 4
    else:
        return 5


# ==========================================
# 3. ФИНАЛЬНЫЙ РАНДОМАЙЗЕР (ИГРОВАЯ ЛОГИКА)
# ==========================================
def roll_final_loot(valid_items, party_level):
    print("\n🎲 Бросаем виртуальные кубики...")

    if random.random() < 0.05:
        return "\n🎲 Выпала странная БЕЗДЕЛУШКА (бросьте d100 по таблице Trinkets)."

    if not valid_items:
        gold_amount = random.randint(10, 50) * party_level
        return f"\n💰 Стоящего лута нет. Вы нашли мешочек с {gold_amount} зм."

    # Взвешенный бросок
    weights = [item['final_score'] for item in valid_items]
    chosen_item = random.choices(valid_items, weights=weights, k=1)[0]
    drop_chance = (chosen_item['final_score'] / sum(weights)) * 100

    # --- ЛОГИКА ОБЪЯСНЕНИЯ (Новое!) ---
    loc_s = chosen_item.get('loc_score', 0)
    party_s = chosen_item.get('party_score', 0)

    if party_s > loc_s + 0.1:
        reason = "Этот предмет идеально подходит способностям вашей группы."
    elif loc_s > party_s + 0.1:
        reason = "Этот трофей выглядит очень уместно в данной локации."
    else:
        reason = "Сбалансированная находка, которая вписывается в окружение и полезна героям."

    result = (
        f"\n✨ НАГРАДА: {chosen_item['name']}\n"
        f"   • Редкость: {chosen_item['rarity']}\n"
        f"   • Тип: {chosen_item['type']}\n"
        f"   • Шанс из пула: {drop_chance:.1f}%\n"
        f"   💬 Комментарий: {reason}\n"  # Наше объяснение
        f"   • Описание: {chosen_item['description'][:200]}..."
    )
    return result


# ==========================================
# 4. ГЛАВНЫЙ КЛАСС СИСТЕМЫ
# ==========================================
class SmartLootGenerator:
    def __init__(self):
        print("Загрузка компонентов ИИ...")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

        with open('dnd_knowledge_base.pkl', 'rb') as f:
            self.kb = pickle.load(f)

        self.kb_embeddings = torch.tensor(np.stack(self.kb['embedding'].values))

        with open('scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)

        self.model = DnDItemRanker(input_size=7)
        self.model.load_state_dict(torch.load('dnd_ranker_weights.pth', weights_only=True))
        self.model.eval()

    def generate_loot(self, location_text, party_text, party_level, story_importance, party_inventory=[]):
        # --- ЭТАП 1: ДВОЙНОЙ ВЕКТОРНЫЙ ПОИСК (Recall) ---
        # Кодируем оба запроса
        loc_emb = self.encoder.encode(location_text, convert_to_tensor=True)
        loc_scores = util.cos_sim(loc_emb, self.kb_embeddings)[0]

        party_emb = self.encoder.encode(party_text, convert_to_tensor=True)
        party_scores = util.cos_sim(party_emb, self.kb_embeddings)[0]

        # Для первичного отбора 50 кандидатов используем среднее
        combined_recall_scores = (loc_scores + party_scores) / 2.0
        top_50_indices = torch.topk(combined_recall_scores, k=50).indices.tolist()

        # --- ЭТАП 2: СБОРКА ПРИЗНАКОВ (Feature Engineering) ---
        features_list = []
        candidates = []
        expected_rarity = get_expected_rarity_for_level(party_level)

        for idx in top_50_indices:
            item = self.kb.iloc[idx]
            l_score = loc_scores[idx].item()
            p_score = party_scores[idx].item()

            # 1. Редкость (Дельта)
            delta = get_rarity_val(item['rarity'], expected_rarity) - expected_rarity

            # 2. РЕАЛЬНЫЕ ДУБЛИКАТЫ
            # Проверяем, есть ли имя предмета в переданном списке party_inventory
            is_duplicate = 1.0 if str(item['name']).lower() in [i.lower() for i in party_inventory] else 0.0

            # 3. КАТЕГОРИЯ ПРЕДМЕТА
            item_type_str = str(item.get('type', 'wondrous item')).lower()
            # Ищем базовый тип, игнорируя приписки типа "(requires attunement)"
            type_id = 0.9  # По умолчанию Wondrous
            for k, v in TYPE_MAP.items():
                if k in item_type_str:
                    type_id = v
                    break

            # 4. СТРОГАЯ СИНЕРГИЯ (Никакой жалости)
            synergy_flag = 0.0
            party_lower = party_text.lower()
            for cls, allowed_types in CLASS_SYNERGY.items():
                if cls in party_lower:
                    # Если хотя бы один класс в партии может использовать этот предмет
                    if any(t in item_type_str for t in allowed_types):
                        synergy_flag = 1.0
                        break

            # ФОРМИРУЕМ НОВЫЙ ВЕКТОР (7 признаков)
            features_list.append([
                l_score, p_score, story_importance, delta, is_duplicate, type_id, synergy_flag
            ])

            # Сохраняем для дебага
            item_dict = item.to_dict()
            item_dict.update({
                'loc_score': l_score,
                'party_score': p_score,
                'delta': delta,
                'is_duplicate': is_duplicate,
                'synergy': synergy_flag
            })
            candidates.append(item_dict)


        # --- ЭТАП 3: MLP ПРЕДСКАЗАНИЕ ---
        X_raw = np.array(features_list)
        X_scaled = self.scaler.transform(X_raw)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        with torch.no_grad():
            predictions = self.model(X_tensor).numpy().flatten()

        # Привязываем скоры ко всем кандидатам
        for i, item in enumerate(candidates):
            item['final_score'] = float(predictions[i])

        # Сортируем ВЕСЬ список до фильтрации, чтобы увидеть лидеров
        candidates.sort(key=lambda x: x['final_score'], reverse=True)

        # ==========================================
        # 🛠️ РАСШИРЕННЫЙ ДЕБАГ-ВЫВОД
        # ==========================================
        print("\n" + "=" * 50)
        print(" 🛠️ DEBUG: ТОП-3 ПРЕДМЕТА ГЛАЗАМИ НЕЙРОСЕТИ")
        print("=" * 50)
        for i in range(min(3, len(candidates))):
            c = candidates[i]
            # Визуальный маркер: прошел предмет порог 0.22 или нет
            status = "✅ ПРОШЕЛ" if c['final_score'] >= 0.22 else "❌ ОТКЛОНЕН"

            print(f"{i + 1}. {c['name']} ({c['rarity']}) -> {status}")
            print(f"   📊 Итоговый MLP Score: {c['final_score']:.3f}")
            print(f"   ├─ Локация: {c['loc_score']:.3f}")
            print(f"   ├─ Партия:  {c['party_score']:.3f}")
            print(f"   └─ Дельта:  {c['delta']}")
            print("-" * 50)
        print("=" * 50 + "\n")

        # --- ЭТАП 4: ПОДГОТОВКА ВАЛИДНОГО ПУЛА ---
        valid_candidates = []
        for i, item in enumerate(candidates):
            item['final_score'] = float(predictions[i])

            # Оставляем только те предметы, которые прошли порог 0.25
            if item['final_score'] >= 0.25:
                valid_candidates.append(item)

        # Сортируем по убыванию вероятности
        valid_candidates.sort(key=lambda x: x['final_score'], reverse=True)

        return valid_candidates


# ==========================================
# 5. ИНТЕРАКТИВНЫЙ ИНТЕРФЕЙС
# ==========================================
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print(" 🐉 УМНЫЙ ГЕНЕРАТОР ЛУТА D&D 5e")
    print("=" * 55)

    generator = SmartLootGenerator()
    print("Система готова к работе!\n")

    while True:
        print("-" * 55)
        command = input("Нажмите [Enter] для генерации или 'q' для выхода: ").strip().lower()
        if command == 'q':
            print("Удачных игр!")
            break

        try:
            party_level = int(input("⚔️ Уровень группы (1-20): "))
            story_importance = float(input("🔥 Важность боя (0.0 - 1.0): "))
        except ValueError:
            print("⚠️ Ошибка ввода чисел.")
            continue

        print("🗺️ ЛОКАЦИЯ (напр: shipwreck, pirate, mutant, poison):")
        loc_input = input("   > ")

        print("🛡️ СОСТАВ ПАРТИИ (напр: Life Cleric, Cavalier Fighter, Assassin Rogue):")
        party_input = input("   > ")

        print("\n🧠 ИИ анализирует двойной контекст...")

        pool = generator.generate_loot(
            location_text=loc_input,
            party_text=party_input,
            party_level=party_level,
            story_importance=story_importance
        )

        print(roll_final_loot(pool, party_level))
        print("-" * 55 + "\n")