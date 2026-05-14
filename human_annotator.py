import os
import logging
import warnings
from models import ITEM_TYPES, CLASS_SYNERGY, get_type_ohe, CLASS_LORE, TERRAIN, ATMOSPHERE, ENEMY_FACTIONS, \
    ENEMY_ACTIONS, build_party_semantics
from generator_data import calculate_target_y

os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['SAFETENSORS_FAST_GPU'] = '1'

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule

console = Console()

import pandas as pd
import numpy as np
import random
import torch
import re
from sentence_transformers import SentenceTransformer, util


def generate_dynamic_scenario():
    loc = f"{random.choice(TERRAIN)}, {random.choice(ATMOSPHERE)}, {random.choice(ENEMY_FACTIONS)}, {random.choice(ENEMY_ACTIONS)}"

    # 🌟 Генерируем партию с подклассами (напр. "Cavalier Fighter")
    party_size = random.randint(3, 5)
    party_members = []
    base_classes_list = list(CLASS_LORE.keys())

    for _ in range(party_size):
        base_cls = random.choice(base_classes_list)
        sub_cls = random.choice(list(CLASS_LORE[base_cls]['subclasses'].keys()))
        party_members.append(f"{sub_cls.capitalize()} {base_cls.capitalize()}")

    party = ", ".join(party_members)
    level = random.randint(1, 20)
    imp = round(random.betavariate(2, 5), 2)

    return {"loc": loc, "party": party, "level": level, "imp": imp}


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


print("Загрузка ИИ-компонентов...")
encoder = SentenceTransformer('all-MiniLM-L6-v2')
kb = pd.read_pickle('dnd_knowledge_base.pkl')
kb_emb = torch.tensor(np.stack(kb['embedding'].values))

GOLD_FILE = 'manual_gold_standard.csv'
base_cols = [
    'item_name', 'location_text', 'party_text',
    'loc_score', 'party_score', 'story_importance',
    'level_rarity_delta', 'is_duplicate', 'synergy_flag', 'target_y'
]
type_cols = [f'type_{t.replace(" ", "_")}' for t in ITEM_TYPES]
cols = base_cols + type_cols

if not os.path.exists(GOLD_FILE):
    pd.DataFrame(columns=cols).to_csv(GOLD_FILE, index=False, sep=';')

print("\n" + "=" * 60)
print(" 🤖 АННОТАТОР ДАННЫХ 🤖")
print("=" * 60)

cheat_sheet = """
[bold red]❌ КРИТИЧЕСКИЕ ШТРАФЫ (1 - 2 балла)[/bold red]
• DELTA >= +2 
• DELTA == +1 при IMP < 0.8
• MAX(LOC, PTY) < 0.15 
• SYNERGY == NO 

[bold yellow]⚖️ НОРМАЛЬНЫЙ ЛУТ (3 - 6 баллов)[/bold yellow]
• Расходники (Potion, Scroll)
• DELTA == 0, но скоры (LOC/PTY) средние (~0.2 - 0.3)
• DELTA < 0 (Слабый лут) в рядовом бою (IMP < 0.4)

[bold green]🌟 ИДЕАЛЬНЫЙ ЛУТ (7 - 10 баллов)[/bold green]
• Высокие скоры LOC/PTY (> 0.4) при DELTA == 0
• DELTA == +1 при IMP > 0.79
"""

console.print(Panel(
    cheat_sheet.strip(),
    title="[bold white]📜 ШПАРГАЛКА (ОРИЕНТИРУЙСЯ ТОЛЬКО НА ЦИФРЫ)[/bold white]",
    border_style="cyan",
    expand=False
))

print("\nНажми Enter, чтобы начать...")
input()

while True:
    scen = generate_dynamic_scenario()

    semantic_party, found_base_classes = build_party_semantics(scen['party'])

    l_emb = encoder.encode(scen['loc'], convert_to_tensor=True)
    p_emb = encoder.encode(semantic_party, convert_to_tensor=True)

    l_scores = util.cos_sim(l_emb, kb_emb)[0]
    p_scores = util.cos_sim(p_emb, kb_emb)[0]

    combined = (l_scores + p_scores) / 2.0
    idx = torch.topk(combined, k=random.randint(1, 15)).indices[-1].item() if random.random() > 0.2 else random.randint(
        0, len(kb) - 1)
    item = kb.iloc[idx]

    l_s = round(l_scores[idx].item(), 4)
    p_s = round(p_scores[idx].item(), 4)
    exp_r = get_expected_rarity(scen['level'])
    delta = get_rarity_val(item['rarity'], exp_r) - exp_r
    is_dup = 0.0

    i_type = str(item.get('type', 'wondrous item')).lower()
    type_ohe = get_type_ohe(i_type)

    syn = 0.0
    for cls, allowed in CLASS_SYNERGY.items():
        if cls in found_base_classes and any(at in i_type for at in allowed):
            syn = 1.0;
            break

    synth_target = calculate_target_y(l_s, p_s, scen['imp'], delta, is_dup, syn, i_type)
    suggested_ans = int(round(synth_target * 9)) + 1

    math_table = Table.grid(padding=(0, 4))
    math_table.add_row(
        f"🎯 [bold cyan]LOC SCORE:[/bold cyan]  [bold white]{l_s:<5}[/bold white]",
        f"🛡️ [bold cyan]PTY SCORE:[/bold cyan]  [bold white]{p_s:<5}[/bold white]"
    )

    syn_color = "[bold green]YES[/bold green]" if syn > 0 else "[bold red]NO[/bold red]"
    delta_color = "[bold red]" if delta >= 2 else ("[bold green]" if delta == 0 else "[bold yellow]")

    math_table.add_row(
        f"🔥 [bold cyan]IMPORTANCE:[/bold cyan] [bold white]{scen['imp']:<5}[/bold white]",
        f"⚖️ [bold cyan]DELTA:[/bold cyan]      {delta_color}{delta:<5}[/]"
    )
    math_table.add_row(
        f"💎 [bold cyan]TYPE:[/bold cyan]       [bold white]{i_type[:15]}[/bold white]",
        f"🤝 [bold cyan]SYNERGY:[/bold cyan]    {syn_color}"
    )

    ctx_table = Table.grid(padding=(0, 1))
    ctx_table.add_row("[dim]LOC:[/dim]", f"[dim]{scen['loc']}[/dim]")
    ctx_table.add_row("[dim]PTY:[/dim]", f"[dim]{scen['party']}[/dim]")
    ctx_table.add_row("[dim]ITEM:[/dim]", f"[dim]{item['name']} ({item.get('rarity', 'common')})[/dim]")

    content = Group(
        Panel(math_table, title="[bold white]📡 ВХОДНЫЕ ТЕНЗОРЫ ДЛЯ MLP[/bold white]", border_style="green"),
        ctx_table
    )

    console.print()
    console.print(Panel(content, title="[bold blue]ОЦЕНКА ПРЕДМЕТА[/bold blue]", border_style="blue", expand=False))

    prompt_text = f"[bold white]Оценка (1-10) [Enter = согласиться с ИИ: [bold green]{suggested_ans}[/bold green]] или 'q': [/bold white]"
    ans = console.input(prompt_text).strip().lower()

    if ans in ['q', 'й', 'quit', 'exit']:
        print("Выход из разметчика. Сохраненные данные в безопасности!")
        break

    if ans == "":
        ans = str(suggested_ans)
        console.print(f"[dim]Принята оценка ИИ: {ans}[/dim]")

    valid_scores = [str(i) for i in range(1, 11)]
    if ans not in valid_scores:
        console.print("[red]⚠️ Ошибка ввода. Пропускаем...[/red]")
        continue

    target_y = round((int(ans) - 1) / 9.0, 4)

    row_dict = {
        'item_name': item['name'], 'location_text': scen['loc'], 'party_text': scen['party'],
        'loc_score': l_s, 'party_score': p_s, 'story_importance': scen['imp'],
        'level_rarity_delta': delta, 'is_duplicate': is_dup, 'synergy_flag': syn, 'target_y': target_y
    }
    for i, t in enumerate(ITEM_TYPES):
        row_dict[f'type_{t.replace(" ", "_")}'] = type_ohe[i]

    pd.DataFrame([row_dict]).to_csv(GOLD_FILE, mode='a', header=False, index=False, sep=';')
    print(f"✅ Сохранено (Таргет: {target_y})")