import os
import logging
import warnings

# 1. Глушим вывод на уровне системных переменных
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['SAFETENSORS_FAST_GPU'] = '1'

# 2. Глушим вывод на уровне Python-логгеров
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.rule import Rule

console = Console()

import pandas as pd
import numpy as np
import random
import torch
import re
import textwrap
from sentence_transformers import SentenceTransformer, util
# ==========================================
# 1. КОНСТАНТЫ И МАТРИЦЫ
# ==========================================
TYPE_MAP = {
    'weapon': 0.1, 'armor': 0.2, 'potion': 0.3, 'ring': 0.4,
    'scroll': 0.5, 'wand': 0.6, 'staff': 0.7, 'rod': 0.8, 'wondrous item': 0.9,
}

CLASS_SYNERGY = {
    'barbarian': ['weapon', 'potion', 'ring', 'wondrous item'],
    'monk': ['weapon', 'potion', 'ring', 'wondrous item'],
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
}

TERRAIN = [
    "dark crypt", "abandoned mine", "city slums", "sewers network",
    "noble estate", "wizard tower", "ancient forest", "frozen tundra",
    "scorching desert", "stinking swamp", "mountain peak", "shipwreck",
    "volcanic crater", "feywild glade", "shadowfell wasteland", "astral plane",
    "underground cavern", "ruined temple", "floating island", "tavern basement"
]

ATMOSPHERE = [
    "thick fog", "heavy rain", "pitch black", "cobwebs and dust",
    "smell of sulfur", "glowing arcane runes", "howling blizzard",
    "eerie silence", "bloodstains", "magical twilight", "overgrown with vines",
    "crumbling walls", "knee-deep mud", "oppressive heat", "toxic fumes"
]

ENEMY_FACTIONS = [
    "bandit highwaymen", "pirate mutineers", "doomsday cultists", "drow assassins",
    "undead horde", "vampire spawn", "necromancer and skeletons",
    "mind flayer colony", "beholder", "yuan-ti abominations", "giant spiders",
    "mimics and ropers", "young red dragon", "hag coven", "fire elementals",
    "abyssal demons", "goblin raiding party", "rogue artificer constructs"
]

ENEMY_ACTIONS = [
    "setting up an ambush", "guarding a locked chest", "conducting a dark ritual",
    "sleeping", "patrolling the area", "fighting a rival group",
    "interrogating a prisoner", "feasting on a corpse", "searching for intruders",
    "repairing their weapons", "hiding in the shadows", "worshipping an idol"
]

CLASSES = [
    "Fighter", "Rogue", "Wizard", "Cleric", "Paladin", "Ranger",
    "Bard", "Warlock", "Sorcerer", "Druid", "Monk", "Barbarian", "Artificer"
]


def generate_dynamic_scenario():
    """Собирает уникальный сценарий из 4 независимых модулей."""
    terrain = random.choice(TERRAIN)
    atmosphere = random.choice(ATMOSPHERE)
    faction = random.choice(ENEMY_FACTIONS)
    action = random.choice(ENEMY_ACTIONS)

    # Итоговая строка локации получается очень насыщенной ключевыми словами
    loc = f"{terrain}, {atmosphere}, {faction}, {action}"

    # Партия от 3 до 5 человек
    party_size = random.randint(3, 5)
    party = ", ".join(random.sample(CLASSES, k=party_size))

    level = random.randint(1, 20)

    # Распределение важности: чаще рядовые бои, реже эпик
    imp = round(random.betavariate(2, 5), 2)

    return {"loc": loc, "party": party, "level": level, "imp": imp}

# ==========================================
# 2. ВСПОМОГАТЕЛЬНАЯ ЛОГИКА
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
    r = str(rarity_str).lower()
    if 'varies' in r: return expected_rarity
    found = []
    if 'artifact' in r: found.append(6)
    if 'legendary' in r: found.append(5)
    if 'very rare' in r: found.append(4); r = r.replace('very rare', '')
    if 'uncommon' in r: found.append(2); r = r.replace('uncommon', '')
    if re.search(r'\brare\b', r): found.append(3)
    if re.search(r'\bcommon\b', r): found.append(1)
    return min(found, key=lambda x: abs(x - expected_rarity)) if found else 1


# ==========================================
# 3. ИНИЦИАЛИЗАЦИЯ И ПОДГОТОВКА ФАЙЛА
# ==========================================

print("Загрузка ИИ-компонентов...")
encoder = SentenceTransformer('all-MiniLM-L6-v2')
kb = pd.read_pickle('dnd_knowledge_base.pkl')
kb_emb = torch.tensor(np.stack(kb['embedding'].values))

GOLD_FILE = 'manual_gold_standard.csv'
# Создаем заголовки для 7 признаков + метаданные
cols = [
    'item_name', 'location_text', 'party_text',
    'loc_score', 'party_score', 'story_importance',
    'level_rarity_delta', 'is_duplicate', 'type_id',
    'synergy_flag', 'target_y'
]

if not os.path.exists(GOLD_FILE):
    pd.DataFrame(columns=cols).to_csv(GOLD_FILE, index=False, sep=';')

# ==========================================
# 4. ОСНОВНОЙ ЦИКЛ РАЗМЕТКИ
# ==========================================

print("\n" + "=" * 60)
print(" 🛡️ РАЗМЕТЧИК DATASET v2.0 (7 FEATURES) 🛡️")
print("=" * 60)

# ==========================================
# 4. ШПАРГАЛКА МАСТЕРА (МАТЕМАТИЧЕСКАЯ)
# ==========================================
cheat_sheet = """
[bold red]❌ БЛОКИРАТОРЫ (Оценка: 1 или 2)[/bold red]
• SYNERGY: NO (Предмет не подходит ни одному классу)
• DELTA > 0 при IMP < 0.60 (Слом баланса экономики)
• MAX(LOC, PTY) < 0.15 (Полностью чужеродный лут)

[bold yellow]🥉 ТИР 1: Обычный бой (IMP: 0.10 - 0.39)[/bold yellow] [dim]Потолок: 6[/dim]
• 6: DELTA == 0  | MAX(LOC, PTY) > 0.30
• 4-5: DELTA == -1 | MAX(LOC, PTY) > 0.20
• 3: DELTA <= -2 ИЛИ MAX(LOC, PTY) < 0.18

[bold cyan]🥈 ТИР 2: Важный бой / Квест (IMP: 0.40 - 0.69)[/bold cyan] [dim]Потолок: 9[/dim]
• 8-9: DELTA == 0  | PTY > 0.30 И LOC > 0.20
• 6-7: DELTA == +1 (ТОЛЬКО при IMP > 0.60) ИЛИ DELTA == 0 со скорами ~0.20
• 4-5: DELTA == -1 (Слабая награда для квеста)
• 3: DELTA <= -2 (Штраф за мусор)

[bold magenta]🥇 ТИР 3: Босс / Финал (IMP: 0.70 - 1.00)[/bold magenta]
• 10: DELTA == +1 | PTY > 0.30 И LOC > 0.25 (God Roll)
• 8-9: DELTA == 0    | PTY > 0.25 И LOC > 0.25
• 6-7: DELTA == -1   (Разочарование)
• 4-5: DELTA <= -2   (Жесткий штраф за мусор с босса)
"""

console.print(Panel(
    cheat_sheet.strip(),
    title="[bold white]📜 ПАМЯТКА ДЛЯ РАЗМЕТКИ (1-10)[/bold white]",
    border_style="green",
    expand=False
))

print("\nНажми Enter, чтобы начать разметку...")
input()

# ==========================================
# 5. ОСНОВНОЙ ЦИКЛ РАЗМЕТКИ
# ==========================================
while True:
    scen = generate_dynamic_scenario()

    # Векторный скоринг
    l_emb = encoder.encode(scen['loc'], convert_to_tensor=True)
    p_emb = encoder.encode(scen['party'], convert_to_tensor=True)
    l_scores = util.cos_sim(l_emb, kb_emb)[0]
    p_scores = util.cos_sim(p_emb, kb_emb)[0]

    # Подбор кандидата
    combined = (l_scores + p_scores) / 2.0
    idx = torch.topk(combined, k=random.randint(1, 15)).indices[-1].item() if random.random() > 0.2 else random.randint(
        0, len(kb) - 1)

    item = kb.iloc[idx]

    # --- СБОР 7 ПРИЗНАКОВ ---
    l_s = round(l_scores[idx].item(), 4)
    p_s = round(p_scores[idx].item(), 4)
    exp_r = get_expected_rarity(scen['level'])
    delta = get_rarity_val(item['rarity'], exp_r) - exp_r
    is_dup = 0  # В разметчике по умолчанию не дубликат

    # Определяем type_id
    i_type = str(item.get('type', 'wondrous item')).lower()
    t_id = 0.9
    for k, v in TYPE_MAP.items():
        if k in i_type: t_id = v; break

    # Определяем synergy_flag
    syn = 0.0
    p_low = scen['party'].lower()
    for cls, allowed in CLASS_SYNERGY.items():
        if cls in p_low and any(at in i_type for at in allowed):
            syn = 1.0;
            break

    # --- ИНСПЕКЦИОННЫЙ ВЫВОД (RICH UI) ---
        # 1. Секция контекста
        ctx_table = Table.grid(padding=(0, 1))
        ctx_table.add_row("🗺️ [bold cyan]LOC:[/bold cyan]", scen['loc'])
        ctx_table.add_row("🛡️ [bold cyan]PTY:[/bold cyan]", scen['party'])
        ctx_table.add_row("📊 [bold cyan]META:[/bold cyan]", f"LVL: {scen['level']} | IMP: {scen['imp']}")

        # 2. Секция предмета
        item_table = Table.grid(padding=(0, 1))
        item_table.add_row("🎁 [bold yellow]ITEM:[/bold yellow]", item['name'])
        item_table.add_row("💎 [bold yellow]TYPE:[/bold yellow]",
                           f"{item.get('type', 'wondrous item')} | RARITY: {item['rarity']}")

        # 3. Секция математики (Взгляд нейросети)
        math_table = Table.grid(padding=(0, 4))
        math_table.add_row(
            f"[dim][1][/dim] LOC_SCORE: [bold]{l_s:<5}[/bold]",
            f"[dim][2][/dim] PTY_SCORE: [bold]{p_s:<5}[/bold]"
        )
        math_table.add_row(
            f"[dim][3][/dim] DELTA:     [bold]{delta:<5}[/bold]",
            f"[dim][4][/dim] TYPE_ID:   [bold]{t_id:<5}[/bold]"
        )

        syn_color = "[bold green]YES[/bold green]" if syn > 0 else "[bold red]NO[/bold red]"
        math_table.add_row(f"[dim][5][/dim] SYNERGY:   {syn_color}", "")

        # Описание предмета (само перенесется по ширине терминала)
        desc_text = Text(str(item.get('description', 'Нет описания.')), justify="left")

        # Собираем всё в одну красивую панель
        content = Group(
            ctx_table,
            Rule(style="blue"),
            item_table,
            Rule(style="blue"),
            desc_text,
            Rule(style="blue"),
            math_table
        )

        console.print()
        console.print(
            Panel(content, title="[bold blue]🛡️ ОЦЕНКА ПРЕДМЕТА[/bold blue]", border_style="blue", expand=False))

        # Цветной ввод (обновлен для 10-балльной шкалы)
        ans = console.input("[bold white]Твоя оценка (1-10) или 'q': [/bold white]").strip().lower()

        # 1. Сначала проверяем на выход (включая русскую 'й')
        if ans in ['q', 'й', 'quit', 'exit']:
            print("Выход из разметчика. Сохраненные данные в безопасности!")
            break

        # 2. Генерируем список валидных ответов: ['1', '2', '3', ..., '10']
        valid_scores = [str(i) for i in range(1, 11)]

        # 3. Проверяем корректность ввода
        if ans not in valid_scores:
            print("[red]⚠️ Ошибка ввода. Нужно ввести число от 1 до 10. Пропускаем...[/red]")
            continue

        # 4. Вычисляем target_y (перевод 1-10 в шкалу 0.0-1.0)
        # 1 -> 0.0 | 5 -> 0.44 | 8 -> 0.77 | 10 -> 1.0
        target_y = round((int(ans) - 1) / 9.0, 4)

        row = pd.DataFrame([{
            'item_name': item['name'], 'location_text': scen['loc'], 'party_text': scen['party'],
            'loc_score': l_s, 'party_score': p_s, 'story_importance': scen['imp'],
            'level_rarity_delta': delta, 'is_duplicate': is_dup, 'type_id': t_id,
            'synergy_flag': syn, 'target_y': target_y
        }])

        row.to_csv(GOLD_FILE, mode='a', header=False, index=False, sep=';')
        print(f"✅ Данные занесены (Твоя оценка: {ans}/10 -> Нейросеть увидит: {target_y})")