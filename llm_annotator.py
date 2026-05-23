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
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

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

API_KEY = os.environ.get("GROQ_API_KEY")
if not API_KEY:
    console.print("[bold red]ОШИБКА: Не найден ключ GROQ_API_KEY в файле .env![/bold red]")
    exit()

client = Groq(api_key=API_KEY, timeout=30.0)
MODEL_NAME = "llama-3.1-8b-instant"


def generate_dynamic_scenario():
    terrain_str = f"{random.choice(TERRAIN)}, {random.choice(PLANES)}" if random.random() < 0.2 else random.choice(
        TERRAIN)
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
    imp = round(random.betavariate(2, 5), 2)

    return {"loc": loc, "party": party, "level": level, "imp": imp}


def normalize_for_llm(ml_score, max_expected=0.45):
    """Переводит косинусное расстояние в понятную для LLM шкалу от 1.0 до 10.0"""
    normalized = (ml_score / max_expected) * 10.0
    return min(10.0, max(0.1, round(normalized, 1)))


def ask_llm_auditor(scen, item, l_s_10, p_s_10, delta, max_retries=5):
    """
    Вызывается ТОЛЬКО если предмет прошел хард-фильтры Python.
    Промпт сфокусирован исключительно на оценке лора и ролевой уместности.
    """

    # === ДИНАМИЧЕСКИЙ КОНТЕКСТ БАЛАНСА И ВАЖНОСТИ ===
    # Здесь мы переносим сложную нелинейную логику из generator_data.py в понятный для ИИ текст
    if delta == 0:
        delta_text = "НОРМА (Идеальный баланс для их уровня)"
    elif delta == 1:
        if scen['imp'] >= 0.70:
            delta_text = "ЧУТЬ СИЛЬНЕЕ (Отличная и заслуженная награда за важный бой)"
        elif scen['imp'] >= 0.50:
            delta_text = "ЧУТЬ СИЛЬНЕЕ (Слегка крутовато для рядового события, но допустимо)"
        else:
            delta_text = "СИЛЬНЕЕ НОРМЫ (Слишком ценная награда для пустякового события! Снижай оценку)"
    else: # delta < 0 (обычно -1 или -2, так как -3 отсекается в Gatekeeper)
        if scen['imp'] >= 0.70:
            delta_text = f"СЛАБЕЕ (На {abs(delta)} тира ниже нормы. Разочаровывающий мусор для эпичного события! Снижай оценку)"
        else:
            delta_text = f"СЛАБЕЕ (На {abs(delta)} тира ниже нормы. Игрокам будет скучновато)"

    system_prompt = """Ты — опытный Dungeon Master. Твоя задача — оценить качество лута (Score от 0.150 до 0.990).
ВНИМАНИЕ: Предмет УЖЕ прошел все системные проверки на баланс и правила. Он полностью легален для выдачи.
Твоя задача — оценить только его СЮЖЕТНУЮ УМЕСТНОСТЬ и РОЛЕВУЮ ПОЛЕЗНОСТЬ.

МЕТРИКИ:
Программа предоставляет тебе баллы "Совпадения" от 1.0 до 10.0. Опирайся на них при оценке.

КРИТЕРИИ ОЦЕНКИ:
- Идеальный лут (0.800 - 0.990): Совпадение с Локацией И Партией высокое (от 7.0 до 10.0). Баланс идеален или заслужен.
- Хороший лут (0.500 - 0.799): Одно из совпадений среднее (4.0 - 6.9). Предмет полезен.
- Средний/Слабый лут (0.150 - 0.499): Совпадения низкие (ниже 4.0) ИЛИ Баланс содержит прямое указание "Снижай оценку".

ОТВЕТ СТРОГО В JSON (где score это число строго от 0.150 до 0.990): {"reasoning": "...", "score": 0.300}"""

    user_prompt = f"""ДАННЫЕ:
- Предмет: {item['name']} (Тип: {item['type']}, Редкость: {item['rarity']})
- Локация: {scen['loc']}
- Состав партии (Ур. {scen['level']}): {scen['party']}
- Баланс предмета: {delta_text}
- Совпадение с Локацией: {l_s_10:.1f} / 10.0
- Совпадение с Партией: {p_s_10:.1f} / 10.0

Выдай оценку в JSON."""

    for attempt in range(max_retries):
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=MODEL_NAME,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            content = chat_completion.choices[0].message.content
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                match = re.search(r'\{.*\}', content, re.DOTALL)
                result = json.loads(match.group(0)) if match else {"score": 0.3, "reasoning": "Parse error"}

            return float(result.get("score", 0.3)), result.get("reasoning", "No reason provided")
        except Exception as e:
            error_msg = str(e).lower()
            if any(code in error_msg for code in ["429", "rate_limit", "timeout", "503", "502", "failed_generation"]):
                wait_time = 15.0 * (attempt + 1)
                console.print(f"[yellow]⏳ Задержка API. Ждем {wait_time} сек... (Попытка {attempt + 1})[/yellow]")
                time.sleep(wait_time)
            else:
                return None, f"Ошибка API: {str(e)}"
    return None, "Timeout."


# --- ЗАГРУЗКА ---
console.print("[bold green]Загрузка ИИ-компонентов и векторной базы...[/bold green]")
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

console.print(f"\n[bold cyan]🚀 ГИБРИДНАЯ РАЗМЕТКА ЗАПУЩЕНА (В базе: {total_annotated})[/bold cyan]")
target_samples = int(console.input("Сколько примеров сгенерировать? (Например, 5000): "))

success_count = 0
with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
    task = progress.add_task("[yellow]Генерация датасета...", total=target_samples)

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

        rarity_str = str(item.get('rarity', 'common')).lower()
        delta = get_rarity_val(rarity_str, exp_r) - exp_r

        is_dup = 1.0 if random.random() < 0.05 else 0.0
        i_type = str(item.get('type', 'wondrous item')).lower()
        type_ohe = get_type_ohe(i_type)

        syn = 0.0
        for cls, allowed in CLASS_SYNERGY.items():
            if cls in found_base_classes and any(at in i_type for at in allowed):
                syn = 1.0;
                break

        # ==========================================
        # АРХИТЕКТУРНЫЙ ПРОРЫВ: PYTHON GATEKEEPER
        # ==========================================
        is_hard_penalty = False
        hard_score = 0.0
        reason = ""
        is_consumable = any(c in i_type for c in ['potion', 'scroll'])

        # 1. Вычисляем "чистое качество" предмета (от 0.0 до 1.0)
        # 0.45 берем как эталонный максимум для косинусного сходства
        norm_l = min(1.0, max(0.0, l_s / 0.45))
        norm_p = min(1.0, max(0.0, p_s / 0.45))
        base_quality = (norm_l + norm_p) / 2.0

        # 2. Штрафуем градиентом: чем лучше предмет, тем ближе он к "потолку" штрафа
        if 'artifact' in rarity_str and scen['imp'] < 0.85:
            is_hard_penalty = True
            hard_score = 0.010 + (0.040 * base_quality)  # Диапазон: от 0.010 до 0.050
            reason = "[PYTHON] Артефакт не подходит по Важности."

        elif 'legendary' in rarity_str and scen['imp'] < 0.75:
            is_hard_penalty = True
            hard_score = 0.020 + (0.040 * base_quality)  # Диапазон: от 0.020 до 0.060
            reason = "[PYTHON] Легендарка не подходит по Важности."

        elif delta >= 2:
            is_hard_penalty = True
            hard_score = 0.050 + (0.080 * base_quality)
            reason = f"[PYTHON] Слом баланса (Дельта {delta})."

        elif delta <= -3:
            is_hard_penalty = True
            hard_score = 0.050 + (0.150 * base_quality)
            reason = f"[PYTHON] Слишком слабо для группы (Дельта {delta})."

        elif syn == 0.0:
            is_hard_penalty = True
            hard_score = 0.050 + (0.100 * base_quality)  # Диапазон: от 0.050 до 0.150
            reason = "[PYTHON] Нет синергии с классами."

        elif is_dup == 1.0 and not is_consumable:
            is_hard_penalty = True
            hard_score = 0.050 + (0.070 * base_quality)  # Диапазон: от 0.050 до 0.120
            reason = "[PYTHON] Бесполезный дубликат."

        # 3. Добавляем микро-шум ТОЛЬКО для дисперсии (чтобы не было жестких линий на графике)
        if is_hard_penalty:
            hard_score += random.gauss(0, 0.005)
            hard_score = max(0.001, min(1.0, hard_score))

        # ==========================================
        # МАРШРУТИЗАЦИЯ (ROUTING)
        # ==========================================
        if is_hard_penalty:
            target_y = round(hard_score, 4)
        else:
            l_s_10 = normalize_for_llm(l_s)
            p_s_10 = normalize_for_llm(p_s)

            score, llm_reason = ask_llm_auditor(scen, item, l_s_10, p_s_10, delta)

            if score is None:
                progress.console.print(f"[red]⚠️ Пропуск: {llm_reason}[/red]")
                time.sleep(3)
                continue

            score = max(0.150, score)

            noise = random.gauss(0, 0.015)
            target_y = max(0.001, min(1.0, round(score + noise, 4)))
            reason = f"[AI] {llm_reason}"
            time.sleep(1.5)  # Пауза только если дергали API

        # Сохранение в базу
        row_dict = {
            'item_name': item['name'], 'location_text': scen['loc'], 'party_text': scen['party'],
            'loc_score': l_s, 'party_score': p_s, 'story_importance': scen['imp'],
            'level_rarity_delta': delta, 'is_duplicate': is_dup, 'synergy_flag': syn, 'target_y': target_y
        }
        for i, t in enumerate(ITEM_TYPES):
            row_dict[f'type_{t.replace(" ", "_")}'] = type_ohe[i]

        pd.DataFrame([row_dict]).to_csv(GOLD_FILE, mode='a', header=False, index=False, sep=';')

        color = "magenta" if "[PYTHON]" in reason else "green"
        progress.console.print(
            f"[dim]Лут:[/dim] [cyan]{item['name'][:25]:<25}[/cyan] | "
            f"[bold white]Y: {target_y:.4f}[/bold white] | "
            f"[dim]L: {l_s:.3f} | P: {p_s:.3f} | D: {delta} | Imp: {scen['imp']:.2f}[/dim]\n"
            f"[italic {color}]{reason}[/italic {color}]\n"
        )

        success_count += 1
        progress.update(task, advance=1)

console.print("\n[bold green]✅ Сессия разметки завершена![/bold green]")