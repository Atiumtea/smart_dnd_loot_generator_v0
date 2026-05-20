import os
import time
import json
import random
import logging
import warnings
import pandas as pd
import numpy as np
import torch
import re
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

from dotenv import load_dotenv
load_dotenv()
API_KEY = os.environ.get("GEMINI_API_KEY")

from models import (
    ITEM_TYPES, CLASS_SYNERGY, get_type_ohe, CLASS_LORE, PLANES, TERRAIN, ATMOSPHERE,
    ENEMY_FACTIONS, ENEMY_ACTIONS, build_party_semantics, get_expected_rarity, get_rarity_val
)

os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['SAFETENSORS_FAST_GPU'] = '1'
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

console = Console()

# --- НАСТРОЙКА GEMINI API ---
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    console.print("[bold red]ОШИБКА: Не найден ключ API![/bold red]")
    exit()

genai.configure(api_key=API_KEY)

model = genai.GenerativeModel(
    'gemini-2.5-flash',
    generation_config={"response_mime_type": "application/json"}
)

def generate_dynamic_scenario():
    terrain_str = f"{random.choice(TERRAIN)}, {random.choice(PLANES)}" if random.random() < 0.2 else random.choice(TERRAIN)
    loc = f"{terrain_str}, {random.choice(ATMOSPHERE)}, {random.choice(ENEMY_FACTIONS)}, {random.choice(ENEMY_ACTIONS)}"
    party_size = random.randint(3, 5)
    party_members = []
    base_classes_list = list(CLASS_LORE.keys())

    for _ in range(party_size):
        base_cls = random.choice(base_classes_list)
        sub_cls = random.choice(list(CLASS_LORE[base_cls]['subclasses'].keys()))
        party_members.append(f"{sub_cls.capitalize()} {base_cls.capitalize()}")

    party = ", ".join(party_members)
    level = random.randint(1, 20)
    imp = round(random.betavariate(2, 5), 2)  # Чаще выдает значения ближе к 0.3-0.5, но бывают и боссы (1.0)

    return {"loc": loc, "party": party, "level": level, "imp": imp}


def ask_llm_auditor(scen, item, l_s, p_s, delta, syn, i_type):
    prompt = f"""
    Ты — ИИ-аудитор датасета для умного генератора лута в D&D 5e.
    Твоя задача — оценить, насколько хорошо этот предмет вписывается в сцену и подходит группе.
    Ответ должен быть ИСКЛЮЧИТЕЛЬНО в формате JSON с двумя полями: "reason" (кратко на русском, почему такая оценка) и "score" (число с плавающей точкой от 0.000 до 1.000).

    ДАННЫЕ:
    - Предмет: {item['name']} ({item.get('rarity', 'common')}, {i_type})
    - Описание: {str(item.get('description', ''))[:400]}...
    - Локация: {scen['loc']}
    - Группа: {scen['party']} (Уровень {scen['level']})
    - Важность битвы: {scen['imp']:.2f} (Где 1.0 - эпичный босс, 0.1 - случайная стычка)

    МАТЕМАТИКА НЕЙРОСЕТИ (ОРИЕНТИРУЙСЯ НА ЭТИ ЦИФРЫ):
    - Loc Score: {l_s} (Насколько предмет лорно подходит локации. >0.3 - отлично, <0.15 - плохо)
    - Party Score: {p_s} (Насколько предмет лорно нужен классам. >0.3 - отлично, <0.15 - плохо)
    - Delta: {delta} (Разница редкости предмета и уровня группы. Важный баланс!)
    - Synergy: {syn} (1.0 = предмет подходит классам, 0.0 = предмет никто не сможет использовать)

    ПОЛНОЕ РУКОВОДСТВО ПО ОЦЕНКЕ (СТРОГИЕ ПРАВИЛА):
    1. КРИТИЧЕСКИЕ ШТРАФЫ (Score от 0.000 до 0.150):
       - Delta >= 2: КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО. Ломает баланс. Оценка ~ 0.000.
       - Delta == 1 И Важность битвы < 0.70: ЗАПРЕЩЕНО. Выдавать лут на тир выше можно только с сильных боссов! Оценка ~ 0.050.
       - Synergy == 0.0: Предмет бесполезен партии (например, латы для магов). Оценка не выше 0.150.
       - Loc Score < 0.15 И Party Score < 0.15: Полностью нерелевантный мусор. Оценка ~ 0.000.
    2. СЛАБЫЙ ИЛИ СИТУАТИВНЫЙ ЛУТ (Score от 0.200 до 0.450):
       - Delta < 0: Предмет слишком слабый для группы. Нормально для легких боев, но плохо для боссов.
       - Loc Score высокий, но Party Score низкий (и наоборот): Предмет подходит только окружению или только партии. Оценка ~ 0.400.
    3. ХОРОШИЙ ЛУТ (Score от 0.500 до 0.750):
       - Loc Score и Party Score средние (0.2 - 0.3), Delta == 0, Synergy == 1.0.
       - Стандартная, адекватная награда за обычный бой.
    4. ИДЕАЛЬНЫЙ ЛУТ (Score от 0.800 до 1.000):
       - Оба скора (Loc и Party) высокие (> 0.35), Delta == 0, Synergy == 1.0.
       - Delta == 1, Важность битвы > 0.85, Synergy == 1.0 (Идеальная награда за сложнейшего босса).
    """

    try:
        response = model.generate_content(prompt)
        result = json.loads(response.text)
        return float(result.get("score", 0.3)), result.get("reason", "No reason provided")
    except Exception as e:
        return None, f"API Error: {str(e)}"

# --- ЗАГРУЗКА ---
console.print("[bold green]Загрузка локальных ИИ-компонентов...[/bold green]")
encoder = SentenceTransformer('all-MiniLM-L6-v2')
kb = pd.read_pickle('dnd_knowledge_base.pkl')
kb_emb = torch.tensor(np.stack(kb['embedding'].values))

GOLD_FILE = 'llm_gold_standard.csv'
base_cols = [
    'item_name', 'location_text', 'party_text',
    'loc_score', 'party_score', 'story_importance',
    'level_rarity_delta', 'is_duplicate', 'synergy_flag', 'target_y'
]
type_cols = [f'type_{t.replace(" ", "_")}' for t in ITEM_TYPES]
cols = base_cols + type_cols

if not os.path.exists(GOLD_FILE):
    pd.DataFrame(columns=cols).to_csv(GOLD_FILE, index=False, sep=';')
    total_annotated = 0
else:
    total_annotated = len(pd.read_csv(GOLD_FILE, sep=';'))

console.print(f"\n[bold cyan]🚀 LLM-АВТОРАЗМЕТКА ЗАПУЩЕНА (В базе уже: {total_annotated})[/bold cyan]")
console.print("Нажмите [bold red]Ctrl+C[/bold red] в любой момент, чтобы остановить процесс.\n")

target_samples = int(console.input("Сколько примеров сгенерировать за эту сессию? (Например, 500): "))

success_count = 0
with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
    task = progress.add_task("[yellow]Генерация и оценка датасета...", total=target_samples)

    while success_count < target_samples:
        scen = generate_dynamic_scenario()
        semantic_party, found_base_classes = build_party_semantics(scen['party'])

        l_emb = encoder.encode(scen['loc'], convert_to_tensor=True)
        p_emb = encoder.encode(semantic_party, convert_to_tensor=True)

        l_scores = util.cos_sim(l_emb, kb_emb)[0]
        p_scores = util.cos_sim(p_emb, kb_emb)[0]

        combined = (l_scores + p_scores) / 2.0
        idx = torch.topk(combined, k=random.randint(1, 15)).indices[
            -1].item() if random.random() > 0.3 else random.randint(0, len(kb) - 1)
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

        # --- ЗАПРОС К GEMINI ---
        score, reason = ask_llm_auditor(scen, item, l_s, p_s, delta, syn, i_type)

        if score is None:
            progress.console.print(f"[red]⚠️ Пропуск: {reason}[/red]")
            time.sleep(5)  # Ждем, если API ругается на лимиты
            continue

        target_y = max(0.0, min(1.0, round(score, 4)))

        row_dict = {
            'item_name': item['name'], 'location_text': scen['loc'], 'party_text': scen['party'],
            'loc_score': l_s, 'party_score': p_s, 'story_importance': scen['imp'],
            'level_rarity_delta': delta, 'is_duplicate': is_dup, 'synergy_flag': syn, 'target_y': target_y
        }
        for i, t in enumerate(ITEM_TYPES):
            row_dict[f'type_{t.replace(" ", "_")}'] = type_ohe[i]

        pd.DataFrame([row_dict]).to_csv(GOLD_FILE, mode='a', header=False, index=False, sep=';')

        progress.console.print(
            f"[dim]Добавлено:[/dim] [cyan]{item['name'][:20]}[/cyan] | "
            f"[white]Оценка: {target_y:.3f}[/white] | "
            f"[italic green]{reason}[/italic green]"
        )

        success_count += 1
        progress.update(task, advance=1)

        # Пауза 4 секунды между запросами (Для бесплатного тарифа Gemini - 15 RPM)
        time.sleep(4)

console.print("\n[bold green]✅ Сессия разметки завершена![/bold green]")