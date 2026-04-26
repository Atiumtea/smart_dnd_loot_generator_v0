import pandas as pd
import numpy as np
import random
import torch
import os
import re
import warnings
import textwrap
from sentence_transformers import SentenceTransformer, util

# Глушим логи
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
warnings.filterwarnings("ignore")

# ==========================================
# 1. КОМБИНАТОРНЫЙ ГЕНЕРАТОР СИТУАЦИЙ (РАСШИРЕННЫЙ)
# ==========================================

BIOMES = [
    # --- Классическое Подземелье и Город ---
    "dark crypt, shadows, dust, cobwebs",
    "abandoned mine, unstable tunnels, darkness, minecarts",
    "city slums, narrow alleys, rain, mud",
    "sewers network, toxic sludge, filth, rats",
    "noble estate, marble floors, hidden vault, luxury",
    "wizard tower, arcane runes, floating books, observatory",

    # --- Дикая Природа ---
    "ancient forest, thick vines, fog, overgrown ruins",
    "frozen tundra, howling blizzard, ancient ice, frost",
    "scorching desert, sandstorm, ancient pyramid, oasis",
    "stinking swamp, quicksand, dead trees, mist",
    "high mountain peak, lightning storms, sheer cliffs",

    # --- Экзотика и Другие Планы (High Fantasy) ---
    "shipwreck, underwater trench, coral reef, high pressure",
    "active volcano crater, magma pools, sulfur, heat",
    "feywild glade, glowing mushrooms, eternal twilight, giant flowers",
    "shadowfell wasteland, decaying matter, despair, monochrome",
    "underdark fungal forest, glowing spores, stalactites, silent",
    "astral plane, floating debris, silver cords, zero gravity",
    "clockwork citadel, grinding gears, steam, brass mechanisms"
]

ENEMIES = [
    # --- Гуманоиды и Преступники ---
    "bandit highwaymen, cutthroats, poison",
    "pirate mutineers, swashbucklers, cannons",
    "cultists of the old god, dark rituals, daggers",
    "drow assassins, poisoned crossbows, darkness",
    "rogue mercenaries, heavily armored veterans",

    # --- Нежить ---
    "undead horde, zombies, skeletons, necromancer",
    "vampire lord, vampire spawn, bats, blood",
    "lich, skeletal mages, phylactery, high magic",
    "banshee, specters, ghosts, necrotic drain",

    # --- Чудовища и Аберрации ---
    "mind flayer colony, intellect devourers, psionics",
    "beholder, disintegration rays, anti-magic cone",
    "yuan-ti abomination, snake cultists, venom",
    "aboleth, mind-controlled thralls, deep water, slime",
    "giant spiders, web traps, phase spiders",
    "mimics, ropers, false treasure, ambush",

    # --- Магические и Планарные существа ---
    "red dragon, kobold minions, fire breath, hoard",
    "hag coven, animated trees, dark curses, witchcraft",
    "elemental prince, fire mephits, chaos, magma",
    "clockwork golems, rogue artificer, constructs",
    "fey tricksters, dryads, illusions, charm",
    "demon lord cultists, hell hounds, abyssal portals"
]

CLASSES = [
    "Fighter", "Rogue", "Wizard", "Cleric", "Paladin", "Ranger",
    "Bard", "Warlock", "Sorcerer", "Druid", "Monk", "Barbarian",
    "Artificer"  # Не забываем про изобретателей!
]


def generate_dynamic_scenario():
    """Создает случайный, но логичный сценарий для D&D."""
    loc = f"{random.choice(BIOMES)}, {random.choice(ENEMIES)}"
    # Выбираем от 3 до 5 случайных классов для партии
    party_size = random.randint(3, 5)
    party = ", ".join(random.sample(CLASSES, k=party_size))
    level = random.randint(1, 20)

    # Распределение важности: чаще рядовые бои, реже эпик
    imp = round(random.betavariate(2, 5), 2)  # Выдаст число с перевесом к 0.2-0.4

    return {"loc": loc, "party": party, "level": level, "imp": imp}


# ==========================================
# 2. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================================
def get_expected_rarity(level):
    if level <= 4:
        return 2
    elif level <= 10:
        return 3
    elif level <= 16:
        return 4
    else:
        return 5


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

# ==========================================
# 3. ИНИЦИАЛИЗАЦИЯ
# ==========================================
print("Загрузка базы и языковой модели...")
encoder = SentenceTransformer('all-MiniLM-L6-v2')
kb = pd.read_pickle('dnd_knowledge_base.pkl')
kb_emb = torch.tensor(np.stack(kb['embedding'].values))

GOLD_FILE = 'manual_gold_standard.csv'
if not os.path.exists(GOLD_FILE):
    df_empty = pd.DataFrame(columns=[
        'item_name', 'location_text', 'party_text',
        'loc_score', 'party_score', 'story_importance', 'level_rarity_delta', 'target_y'
    ])
    df_empty.to_csv(GOLD_FILE, index=False, sep=';')

print("\n" + "=" * 50)
print(" === РЕЖИМ ГЕЙМ-МАСТЕРА (РАЗМЕТЧИК) ===")
print("=" * 50)
print("Оценивайте уместность предмета от 1 до 5.")
print("1 - Руинит баланс / Мусор\n3 - Проходняк, можно дать\n5 - Идеально вписывается!")
print("Введите 'q' для выхода.\n")

# ==========================================
# 4. ЦИКЛ РАЗМЕТКИ
# ==========================================
while True:
    scenario = generate_dynamic_scenario()

    loc_e = encoder.encode(scenario['loc'], convert_to_tensor=True)
    party_e = encoder.encode(scenario['party'], convert_to_tensor=True)

    loc_scores = util.cos_sim(loc_e, kb_emb)[0]
    party_scores = util.cos_sim(party_e, kb_emb)[0]

    combined = (loc_scores + party_scores) / 2.0

    # 70% шанс получить релевантный предмет, 30% шанс получить случайный мусор
    if random.random() > 0.3:
        idx = torch.topk(combined, k=random.randint(1, 20)).indices[-1].item()
    else:
        idx = random.randint(0, len(kb) - 1)

    item = kb.iloc[idx]

    l_score = round(loc_scores[idx].item(), 4)
    p_score = round(party_scores[idx].item(), 4)
    delta = get_rarity_val(item['rarity']) - get_expected_rarity(scenario['level'])

    print("\n" + "-" * 55)
    print(f"🌍 ЛОКАЦИЯ:  {scenario['loc']}")
    print(f"🛡️ ПАРТИЯ:   {scenario['party']} (Ур: {scenario['level']})")
    print(f"🔥 ВАЖНОСТЬ: {scenario['imp']}")
    print("-" * 55)
    print(f"🎁 ПРЕДМЕТ:  {item['name']} | Редкость: {item['rarity']}")
    print(f"   [Дельта редкости: {delta} | Скор локации: {l_score:.2f} | Скор партии: {p_score:.2f}]")

    # Вывод описания (с переносом строк для читаемости)
    desc = str(item.get('description', 'Нет описания.'))
    # Обрезаем описание, если оно слишком длинное
    if len(desc) > 400: desc = desc[:397] + "..."
    print("\n📖 ОПИСАНИЕ:")
    print(textwrap.fill(desc, width=55, initial_indent="    ", subsequent_indent="    "))
    print("-" * 55)

    ans = input("Твоя оценка (1-5) или 'q': ").strip().lower()

    if ans == 'q': break
    if ans not in ['1', '2', '3', '4', '5']:
        print("⚠️ Ошибка ввода. Пропускаем этот предмет.")
        continue

    score_map = {'1': 0.0, '2': 0.25, '3': 0.5, '4': 0.75, '5': 1.0}
    target_y = score_map[ans]

    new_row = pd.DataFrame([{
        'item_name': item['name'],
        'location_text': scenario['loc'],
        'party_text': scenario['party'],
        'loc_score': l_score,
        'party_score': p_score,
        'story_importance': scenario['imp'],
        'level_rarity_delta': delta,
        'target_y': target_y
    }])
    new_row.to_csv(GOLD_FILE, mode='a', header=False, index=False, sep=';')
    print("✅ Успешно записано в золотой стандарт!")