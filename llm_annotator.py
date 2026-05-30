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
    ENEMY_FACTIONS, ENEMY_ACTIONS, build_party_semantics, get_tier_brackets, get_rarity_val, calculate_level_delta
)

os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['SAFETENSORS_FAST_GPU'] = '1'
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

console = Console()

# --- ВЫБОР API КЛЮЧА И НАСТРОЙКА АВТОПЕРЕКЛЮЧЕНИЯ ---
console.print()
console.print("[bold yellow]Доступные ключи API:[/bold yellow]")
for i in range(1, 8):
    console.print(f"{i}: GROQ_API_KEY_{i}")
console.print()

console.print("[bold green]Выберите ключ (введите 1-7) или напишите 'auto' для автопереключения:[/bold green]", end=" ")
key_choice = console.input("").strip().lower()

auto_mode = False
current_key_idx = 1
client = None

def init_client(key_num):
    env_key_name = f"GROQ_API_KEY_{key_num}"
    api_key = os.environ.get(env_key_name)
    if not api_key:
        return None, env_key_name
    return Groq(api_key=api_key, timeout=30.0), env_key_name

if key_choice == 'auto':
    auto_mode = True
    client, env_key_name = init_client(current_key_idx)
    if not client:
        console.print(f"[bold red]ОШИБКА: Стартовый ключ {env_key_name} не найден в .env![/bold red]")
        exit()
    console.print(f"\n[bold green]✅ Включен АВТОРЕЖИМ. Старт с: {env_key_name}[/bold green]\n")
elif key_choice in ['1', '2', '3', '4', '5', '6', '7']:
    current_key_idx = int(key_choice)
    client, env_key_name = init_client(current_key_idx)
    if not client:
        console.print(f"[bold red]ОШИБКА: Не найден ключ {env_key_name} в файле .env![/bold red]")
        exit()
    console.print(f"\n[bold green]✅ Подключен ключ: {env_key_name}[/bold green]\n")
else:
    console.print("[bold red]Неверный ввод. Выбран GROQ_API_KEY_1 без автопереключения.[/bold red]")
    client, env_key_name = init_client(1)
    if not client:
        console.print("[bold red]ОШИБКА: Ключ GROQ_API_KEY_1 отсутствует в .env![/bold red]")
        exit()

def rotate_api_key(attempts=0):
    global client, current_key_idx
    if attempts >= 7:
        console.print("[bold red]❌ Все 7 ключей невалидны или отсутствуют в .env![/bold red]")
        return False

    current_key_idx += 1
    if current_key_idx > 7:
        current_key_idx = 1
        console.print("[yellow]⏳ Прошли все 7 ключей. Возвращаемся к 1-му. Ждем 5 сек...[/yellow]")
        time.sleep(5)

    new_client, env_key_name = init_client(current_key_idx)
    if new_client:
        client = new_client
        console.print(f"\n[bold magenta]🔄 Лимит! Авто-переключение на: {env_key_name}[/bold magenta]")
        return True
    else:
        # Если ключ пропущен в .env, ищем следующий
        return rotate_api_key(attempts + 1)

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

def ask_llm_auditor(scen, item, delta, max_retries=15):
    if delta == 0:
        delta_text = "NORMAL (Ideal balance for their level)"
    elif delta > 0:
        if delta <= 4:
            if scen['imp'] >= 0.70:
                delta_text = f"SLIGHTLY STRONGER ({delta} levels higher. Well-deserved reward for a major battle)"
            else:
                delta_text = f"STRONGER THAN NORMAL ({delta} levels higher. Too valuable for a minor event! STRICTLY REDUCE SCORE)"
        else:
            delta_text = f"TOO STRONG ({delta} levels higher. STRICTLY REDUCE SCORE)"
    else:
        abs_d = abs(delta)
        if scen['imp'] >= 0.70:
            delta_text = f"WEAKER ({abs_d} levels lower. Disappointing loot for a major event! STRICTLY REDUCE SCORE)"
        else:
            delta_text = f"WEAKER ({abs_d} levels lower. Players might find it boring)"

    system_prompt = """You are a Data Auditor and an experienced Dungeon Master for D&D 5e.
ATTENTION: The item has ALREADY passed strict systemic checks for game-breaking issues. It is legal to drop.
Your task is to provide a final evaluation (Score from 0.150 to 0.990) based ONLY on NARRATIVE APPROPRIATENESS, ROLEPLAY UTILITY, and BALANCE.

METRICS:
The program provides you with "Match" scores from 1.0 to 10.0. Rely on them.

EVALUATION RULES (STRICT):
1. Read the "Item Balance" field carefully. If it says "STRICTLY REDUCE SCORE" in all caps, you MUST penalize the item and output a Score between 0.150 and 0.450, even if Location and Party matches are 10.0!
2. Ideal loot (0.800 - 0.990): Location AND Party matches >= 7.0. Balance is NORMAL or "Well-deserved reward".
3. Good loot (0.500 - 0.799): One of the matches is >= 5.0. Balance is acceptable.
4. Average/Weak loot (0.150 - 0.499): Matches are < 5.0 OR Balance requires a score reduction.

OUTPUT STRICTLY IN JSON FORMAT:
{
  "loc_analysis": "Brief analysis of how the item fits the location",
  "party_analysis": "Brief analysis of how the item fits the party classes",
  "balance_check": "Analyze the Balance field (if it says reduce score, confirm the penalty here)",
  "score": <float_number_from_0.150_to_0.990>
}"""

    user_prompt = f"""DATA:
- Item: {item['name']} (Type: {item['type']}, Rarity: {item['rarity']})
- Location: {scen['loc']}
- Party Composition (Level {scen['level']}): {scen['party']}
- Item Balance: {delta_text}

Perform the analysis and output JSON."""

    for attempt in range(max_retries):
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=MODEL_NAME,
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            content = chat_completion.choices[0].message.content

            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                match = re.search(r'\{.*\}', content, re.DOTALL)
                result = json.loads(match.group(0)) if match else {"score": 0.3}

            score = float(result.get("score", 0.3))
            llm_reason = (
                f"Loc: {result.get('loc_analysis', '')} | "
                f"Party: {result.get('party_analysis', '')} | "
                f"Bal: {result.get('balance_check', 'No data')}"
            )
            return score, llm_reason

        except Exception as e:
            error_msg = str(e).lower()
            if any(code in error_msg for code in ["429", "rate_limit", "timeout", "503", "502", "failed_generation", "limit"]):
                if auto_mode:
                    if rotate_api_key():
                        time.sleep(0.5)
                        continue
                    else:
                        return None, "Не удалось переключить ключ API."
                else:
                    wait_time = 15.0 * (attempt + 1)
                    console.print(f"[yellow]⏳ Задержка API. Ждем {wait_time} сек... (Попытка {attempt + 1})[/yellow]")
                    time.sleep(wait_time)
            else:
                return None, f"Ошибка API: {str(e)}"

    return None, "Превышен лимит таймаутов API."


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

console.print()
console.print(f"[bold cyan]🚀 ГИБРИДНАЯ РАЗМЕТКА ЗАПУЩЕНА (В базе: {total_annotated})[/bold cyan]")

while True:
    try:
        user_input = console.input("[bold yellow]Сколько примеров сгенерировать? (Например, 5000): [/bold yellow]")
        target_samples = int(user_input)
        break
    except ValueError:
        console.print("[bold red]Пожалуйста, введите корректное число.[/bold red]")

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
        idx = torch.topk(combined, k=random.randint(1, 15)).indices[-1].item() if random.random() > 0.3 else random.randint(0, len(kb) - 1)
        item = kb.iloc[idx]

        l_s = round(l_scores[idx].item(), 4)
        p_s = round(p_scores[idx].item(), 4)

        rarity_str = str(item.get('rarity', 'common')).lower()
        rarity_val = get_rarity_val(rarity_str, scen['level'])

        delta = calculate_level_delta(rarity_val, scen['level'])

        is_dup = 1.0 if random.random() < 0.05 else 0.0
        i_type = str(item.get('type', 'wondrous item')).lower()
        type_ohe = get_type_ohe(i_type)

        syn = 0.0
        for cls, allowed in CLASS_SYNERGY.items():
            if cls in found_base_classes and any(at in i_type for at in allowed):
                syn = 1.0;
                break

        # ==========================================
        # НЕПРЕРЫВНЫЙ PYTHON GATEKEEPER И НОРМАЛИЗАЦИЯ
        # ==========================================
        penalty_multiplier = 1.0
        reason_parts = []
        force_python = False
        is_consumable = any(c in i_type for c in ['potion', 'scroll', 'arrow', 'bolt', 'dart'])

        # Нормализация сырых скоров к диапазону 0.0 - 1.0
        eff_l = max(0.0, l_s - 0.12) / (0.45 - 0.12)
        eff_p = max(0.0, p_s - 0.12) / (0.45 - 0.12)

        norm_l = min(1.0, max(0.0, eff_l))
        norm_p = min(1.0, max(0.0, eff_p))

        base_quality = (norm_l + norm_p) / 2.0

        # 1. ФУНДАМЕНТАЛЬНЫЙ ФИЛЬТР РЕЛЕВАНТНОСТИ
        if base_quality < 0.22:
            penalty_multiplier *= 0.20
            reason_parts.append(f"Очень низкая релевантность (L+P: {base_quality:.2f})")
            force_python = True

        # 2. СЮЖЕТНЫЕ ОГРАНИЧЕНИЯ
        if 'artifact' in rarity_str and scen['imp'] < 0.85:
            penalty_multiplier *= 0.33
            reason_parts.append(f"Артефакт в рядовом бою (Imp {scen['imp']:.2f})")

        elif 'legendary' in rarity_str and scen['imp'] < 0.70:
            penalty_multiplier *= 0.44
            reason_parts.append(f"Легендарка в рядовом бою (Imp {scen['imp']:.2f})")

        elif 'very rare' in rarity_str and scen['imp'] < 0.50:
            penalty_multiplier *= 0.56
            reason_parts.append(f"Очень редкий лут не к месту (Imp {scen['imp']:.2f})")

        # 3. БАЛАНС УРОВНЕЙ
        if delta > 0:
            severity_base = (delta / 2.0) ** 2.5
            importance_forgiveness = max(0.1, 1.1 - scen['imp'])
            severity = severity_base * importance_forgiveness
            multiplier = 1.0 / (1.0 + severity)
            penalty_multiplier *= multiplier
            reason_parts.append(f"Рано на {delta} ур.")
            if delta > 3:
                force_python = True

        elif delta < 0:
            abs_d = abs(delta)
            normalized_abs_d = abs_d / 4.0
            severity = (normalized_abs_d ** 2.5) * (0.75 + scen['imp'])
            multiplier = 1.0 / (1.0 + severity)
            penalty_multiplier *= multiplier
            reason_parts.append(f"Поздно на {abs_d} ур.")
            if abs_d > 4:
                force_python = True

        # 4. СИНЕРГИЯ И ДУБЛИКАТЫ
        if syn == 0.0:
            penalty_multiplier *= 0.35
            reason_parts.append("Нет синергии")

        if is_dup == 1.0 and not is_consumable:
            penalty_multiplier *= 0.20
            reason_parts.append("Дубликат")

        # ==========================================
        # МАРШРУТИЗАЦИЯ (ROUTING)
        # ==========================================
        is_hard_penalty = force_python or (penalty_multiplier < 0.50)

        if is_hard_penalty:
            hard_score = base_quality * penalty_multiplier
            hard_score += random.gauss(0, 0.005)
            target_y = round(max(0.001, min(0.999, hard_score)), 4)
            reason = "[PYTHON] " + " + ".join(reason_parts) if reason_parts else "[PYTHON] Unknown penalty"
        else:
            score, llm_reason = ask_llm_auditor(scen, item, delta)

            if score is None:
                progress.console.print(f"[red]⚠️ Пропуск: {llm_reason}[/red]")
                time.sleep(3)
                continue

            weighted_raw = (norm_l * 0.55) + (norm_p * 0.45)
            raw_target = (score * 0.75) + (weighted_raw * 0.25)
            noise = random.gauss(0, 0.015)

            target_y = round(max(0.150, min(0.999, raw_target + noise)), 4)
            reason = f"[AI] {llm_reason}"
            time.sleep(1.5) # Штатная задержка между успешными запросами

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