import os
import sys
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
for i in range(1, 11):
    console.print(f"{i}: GROQ_API_KEY_{i}")
console.print()

console.print("[bold green]Выберите ключ (введите 1-10) или напишите 'auto' для автопереключения:[/bold green]", end=" ")
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
elif key_choice in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']:
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
    if attempts >= 10:
        console.print("[bold red]❌ Все 10 ключей невалидны или отсутствуют в .env![/bold red]")
        return False

    current_key_idx += 1
    if current_key_idx > 10:
        current_key_idx = 1
        console.print("[yellow]⏳ Прошли все 10 ключей. Возвращаемся к 1-му. Ждем 5 сек...[/yellow]")
        time.sleep(5)

    new_client, env_key_name = init_client(current_key_idx)
    if new_client:
        client = new_client
        console.print(f"\n[yellow]🔄 Лимит! Авто-переключение на: {env_key_name}[/yellow]")
        return True
    else:
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

def ask_llm_auditor(scen, item, gatekeeper_log, synergy_count, party_size, max_retries=25):

    system_prompt = """You are a Data Auditor and an experienced Dungeon Master for D&D 5e.
Your task is to provide a final evaluation (Score from 0.150 to 0.990) based on NARRATIVE APPROPRIATENESS, ROLEPLAY UTILITY, and GATEKEEPER NOTES.

EVALUATION RULES (STRICT):
1. READ THE "GATEKEEPER NOTES" CAREFULLY. These are mechanical checks performed by the system.
2. If Gatekeeper Notes say "No mechanical penalties. Standard balance maintained.", evaluate the narrative OBJECTIVELY. Do not inflate the score. A score of 0.500-0.700 is perfectly fine.
3. If Gatekeeper Notes say "Strong contextual match. Above average utility.", the item is highly relevant but has minor flaws. Award a score between 0.700 and 0.850.
4. If Gatekeeper Notes say "Mechanically perfect. Exceptional contextual fit.", the item is statistically ideal. You are highly encouraged to award a top-tier score (0.850 - 0.990).
5. If Gatekeeper Notes contain warnings (e.g., "Too early", "Weak context bottleneck", "Duplicate"), you MUST penalize the score heavily.
6. Output your analysis and the final float score in JSON format.

OUTPUT STRICTLY IN JSON FORMAT:
{
  "loc_analysis": "Brief analysis of how the item fits the location",
  "party_analysis": "Brief analysis of how the item fits the party classes",
  "gatekeeper_integration": "Acknowledge the Gatekeeper Notes and state how they impacted your score",
  "score": <float_number_from_0.150_to_0.990>
}"""

    user_prompt = f"""DATA:
- Item: {item['name']} (Type: {item['type']}, Rarity: {item['rarity']})
- Location: {scen['loc']}
- Party Composition (Level {scen['level']}): {scen['party']}
- Class Synergy: {synergy_count} out of {party_size} party members can optimally use this item.
- Gatekeeper Notes (Mechanical Penalties): {gatekeeper_log}

Perform the analysis and output JSON."""

    rotation_count = 0

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
                f"Gate: {result.get('gatekeeper_integration', 'No data')}"
            )
            return score, llm_reason

        except Exception as e:
            error_msg = str(e).lower()

            # 1. ЛИМИТЫ API
            if any(code in error_msg for code in ["429", "rate_limit", "limit", "too_many"]):
                if auto_mode:
                    rotation_count += 1
                    if rotation_count >= 10:
                        console.print("\n[bold red]🚨 ВСЕ 10 КЛЮЧЕЙ ИСЧЕРПАЛИ ЛИМИТЫ![/bold red]")
                        console.print(
                            "[bold red]🛑 Генератор принудительно завершает работу. Текущий прогресс сохранен в CSV.[/bold red]\n")
                        sys.exit(1)
                    else:
                        rotate_api_key()
                        time.sleep(1)
                        continue
                else:
                    wait_time = 15.0 * (attempt + 1)
                    console.print(f"[yellow]⏳ Лимит ключа. Ждем {wait_time} сек...[/yellow]")
                    time.sleep(wait_time)
                    continue

            # 2. ПРОБЛЕМЫ С СЕТЬЮ / СЕРВЕРОМ GROQ
            elif any(code in error_msg for code in
                     ["timeout", "503", "502", "500", "connection", "connect", "network", "peer"]):
                console.print(f"[yellow]⏳ Сбой сети/сервера Groq. Ждем 15 сек... (Попытка {attempt + 1})[/yellow]")
                time.sleep(15)
                continue

            # 3. НЕИЗВЕСТНЫЕ ОШИБКИ
            else:
                console.print(
                    f"[yellow]⚠️ Неизвестная ошибка: {error_msg}. Ждем 10 сек... (Попытка {attempt + 1})[/yellow]")
                time.sleep(10)
                continue

    return None, "Timeout limit exceeded."

# --- ЗАГРУЗКА ---
console.print("[bold green]Загрузка ИИ-компонентов и векторной базы...[/bold green]")
encoder = SentenceTransformer('all-MiniLM-L6-v2')
kb = pd.read_pickle('dnd_knowledge_base.pkl')
kb_emb = torch.tensor(np.stack(kb['embedding'].values))

GOLD_FILE = 'llm_gold_standard.csv'

base_cols = [
    'item_name', 'location_text', 'party_text',
    'loc_score', 'party_score', 'story_importance', 'rarity_val',
    'level_rarity_delta', 'is_consumable', 'is_duplicate', 'synergy_density', 'target_y'
]
type_cols = [f'type_{t.replace(" ", "_")}' for t in ITEM_TYPES]
cols = base_cols + type_cols

if os.path.exists(GOLD_FILE):
    existing_df = pd.read_csv(GOLD_FILE, sep=';')
    if 'synergy_density' not in existing_df.columns:
        console.print("[bold yellow]⚠️ Обнаружена старая структура датасета. Файл будет перезаписан.[/bold yellow]")
        pd.DataFrame(columns=cols).to_csv(GOLD_FILE, index=False, sep=';')
        total_annotated = 0
    else:
        total_annotated = len(existing_df)
else:
    pd.DataFrame(columns=cols).to_csv(GOLD_FILE, index=False, sep=';')
    total_annotated = 0

console.print()
console.print(f"[bold cyan]🚀 ГИБРИДНАЯ РАЗМЕТКА ЗАПУЩЕНА (В базе: {total_annotated})[/bold cyan]")

remaining_to_goal = max(1000, 25000 - total_annotated)

while True:
    try:
        user_input = console.input(
            f"[bold yellow]Сколько примеров сгенерировать? (Нажмите Enter для {remaining_to_goal}): [/bold yellow]").strip()

        if not user_input:
            target_samples = remaining_to_goal
            console.print(f"[dim]✅ Принято: {target_samples}[/dim]")
            break

        target_samples = int(user_input)
        break
    except ValueError:
        console.print("[bold red]❌ Пожалуйста, введите число или просто нажмите Enter.[/bold red]")

success_count = 0
with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
    task = progress.add_task(f"[yellow]Генерация датасета (Ключ {current_key_idx})...", total=target_samples)

    while success_count < target_samples:
        scen = generate_dynamic_scenario()
        semantic_party, found_base_classes = build_party_semantics(scen['party'])
        party_size = len(found_base_classes)

        l_emb = encoder.encode(scen['loc'], convert_to_tensor=True)
        p_emb = encoder.encode(semantic_party, convert_to_tensor=True)

        l_scores = util.cos_sim(l_emb, kb_emb)[0]
        p_scores = util.cos_sim(p_emb, kb_emb)[0]

        combined = (l_scores + p_scores) / 2.0
        rand_val = random.random()

        if rand_val < 0.40:
            idx = torch.topk(combined, k=random.randint(1, 100)).indices[-1].item()
        elif rand_val < 0.55:
            idx = torch.topk(l_scores, k=random.randint(1, 30)).indices[-1].item()
        elif rand_val < 0.80:
            idx = torch.topk(combined, k=random.randint(100, min(500, len(kb)))).indices[-1].item()
        else:
            idx = random.randint(0, len(kb) - 1)

        item = kb.iloc[idx]

        l_s = round(l_scores[idx].item(), 4)
        p_s = round(p_scores[idx].item(), 4)

        rarity_str = str(item.get('rarity', 'common')).lower()
        rarity_val = float(get_rarity_val(rarity_str, scen['level']))
        delta = calculate_level_delta(rarity_val, scen['level'])

        i_type = str(item.get('type', 'wondrous item')).lower()
        type_ohe = get_type_ohe(i_type)

        is_consumable = 1.0 if any(c in i_type for c in ['potion', 'scroll', 'arrow', 'bolt', 'dart']) else 0.0
        is_dup = 1.0 if random.random() < 0.05 else 0.0

        synergy_count = 0
        for cls in found_base_classes:
            if any(at in i_type for at in CLASS_SYNERGY.get(cls, [])):
                synergy_count += 1
        syn_density = synergy_count / max(1, party_size)

        # ==========================================
        # ГЛАДКИЙ PYTHON GATEKEEPER (SMOOTH PENALTIES)
        # ==========================================
        penalty_multiplier = 1.0
        reason_parts = []
        force_python = False

        safe_l_s = max(0.0, l_s)
        norm_l = min(1.0, (safe_l_s / 0.45) ** 0.8)
        safe_p_s = max(0.0, p_s)
        norm_p = min(1.0, (safe_p_s / 0.45) ** 0.8)
        base_quality = (norm_l + norm_p) / 2.0

        if base_quality < 0.35:
            penalty_multiplier *= (max(0.01, base_quality) / 0.35) ** 2
            reason_parts.append(f"Low base relevance ({base_quality:.2f})")

        bottleneck = min(norm_l, norm_p)
        if bottleneck < 0.30:
            bottleneck_penalty = (max(0.001, bottleneck) / 0.30) ** 1.3
            penalty_multiplier *= bottleneck_penalty
            if bottleneck < 0.10:
                force_python = True
                reason_parts.append(f"Critical context mismatch (Min={bottleneck:.2f})")
            else:
                reason_parts.append(f"Weak context bottleneck (Min={bottleneck:.2f})")

        expected_rarity = 1.0 + scen['imp'] * 4.0 #заменить на 5
        rarity_diff = rarity_val - expected_rarity

        if rarity_val >= 6.0 and scen['imp'] >= 0.85: # вырезать
            rarity_diff = 0.0

        if rarity_diff > 0.5:
            severity = (rarity_diff / 1.5) ** 2
            penalty_multiplier *= 1.0 / (1.0 + severity)
            reason_parts.append(f"Too rare (Item: {rarity_val}, Expected: {expected_rarity:.1f})")

        if delta > 0:
            severity = (delta ** 1.5) * max(0.05, 0.85 - scen['imp']) * 4.0
            penalty_multiplier *= 1.0 / (1.0 + severity)
            reason_parts.append(f"Too early by {delta} levels")
        elif delta < 0:
            severity = ((abs(delta) / 4.0) ** 2) * (0.5 + scen['imp'])
            penalty_multiplier *= 1.0 / (1.0 + severity)
            reason_parts.append(f"Too late by {abs(delta)} levels")

        if syn_density == 0.0:
            penalty_multiplier *= 0.65
            reason_parts.append("Zero class synergy")
        elif syn_density < 0.30:
            penalty_multiplier *= 0.85
            reason_parts.append("Low class synergy")

        if is_dup == 1.0 and is_consumable == 0.0:
            penalty_multiplier *= 0.65
            reason_parts.append("Duplicate non-consumable equipment (Loot bloat - apply penalty)")

        # ==========================================
        # МАРШРУТИЗАЦИЯ И БАЛАНСИРОВКА
        # ==========================================
        is_hard_penalty = force_python or (penalty_multiplier < 0.40)

        if not reason_parts:
            if norm_l >= 0.64 and bottleneck >= 0.44 and base_quality >= 0.58:
                gatekeeper_log = "Mechanically perfect. Exceptional contextual fit."
            elif base_quality >= 0.50 or (norm_l >= 0.55 and bottleneck >= 0.35):
                gatekeeper_log = "Strong contextual match. Above average utility."
            else:
                gatekeeper_log = "No mechanical penalties. Standard balance maintained."
        else:
            gatekeeper_log = " | ".join(reason_parts)

        if is_hard_penalty:
            if random.random() > 0.66:
                continue

            hard_score = base_quality * penalty_multiplier + random.gauss(0, 0.005)
            target_y = round(max(0.001, min(0.999, hard_score)), 4)
            final_output_text = f"[magenta][Gatekeeper]: {gatekeeper_log}[/magenta]"
        else:
            score, llm_reason = ask_llm_auditor(scen, item, gatekeeper_log, synergy_count, party_size)

            if score is None:
                progress.console.print(f"[red]⚠️ Пропуск: {llm_reason}[/red]")
                if "rotation failed" in llm_reason:
                    progress.console.print(
                        "[bold red]🚨 Все ключи мертвы. Спим 30 минут до сброса лимитов...[/bold red]")
                    time.sleep(1800)
                else:
                    time.sleep(3)
                continue

            weighted_raw = (norm_l * 0.55) + (norm_p * 0.45)

            max_allowed_score = min(0.999, weighted_raw + 0.25)

            raw_target = (score * 0.70) + (weighted_raw * 0.30)

            capped_target = min(raw_target, max_allowed_score)
            target_y = round(max(0.150, min(0.999, capped_target + random.gauss(0, 0.015))), 4)

            final_output_text = f"[magenta][Gatekeeper]: {gatekeeper_log}[/magenta]\n[green][LLM]: {llm_reason}[/green]"
            time.sleep(1.5)

        row_dict = {
            'item_name': item['name'], 'location_text': scen['loc'], 'party_text': scen['party'],
            'loc_score': l_s, 'party_score': p_s, 'story_importance': scen['imp'],
            'rarity_val': rarity_val, 'level_rarity_delta': delta, 'is_consumable': is_consumable,
            'is_duplicate': is_dup, 'synergy_density': syn_density, 'target_y': target_y
        }
        for i, t in enumerate(ITEM_TYPES):
            row_dict[f'type_{t.replace(" ", "_")}'] = type_ohe[i]

        pd.DataFrame([row_dict]).to_csv(GOLD_FILE, mode='a', header=False, index=False, sep=';')

        progress.console.print(
            f"[dim]Лут:[/dim] [cyan]{item['name'][:25]:<25}[/cyan] | "
            f"[bold white]Y: {target_y:.4f}[/bold white] | "
            f"[dim]L: {l_s:.3f} (N:{norm_l:.2f}) | P: {p_s:.3f} (N:{norm_p:.2f}) | "
            f"D: {delta} | Syn: {syn_density:.2f} | Dup: {int(is_dup)}[/dim] | Imp: {scen['imp']:.2f}\n"
            f"{final_output_text}\n"
        )

        success_count += 1
        progress.update(task, advance=1, description=f"[yellow]Генерация датасета (Ключ {current_key_idx})...")

console.print("\n[bold green]✅ Сессия разметки завершена![/bold green]")