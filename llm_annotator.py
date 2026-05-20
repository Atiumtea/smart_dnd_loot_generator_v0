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
from rich.panel import Panel

# НОВАЯ БИБЛИОТЕКА
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

# --- НАСТРОЙКА GROQ API ---
API_KEY = os.environ.get("GROQ_API_KEY")
if not API_KEY:
    console.print("[bold red]ОШИБКА: Не найден ключ GROQ_API_KEY в файле .env![/bold red]")
    exit()

# Инициализируем клиента Groq
client = Groq(api_key=API_KEY)
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


def ask_llm_auditor(scen, item, l_s, p_s, delta, syn, i_type, is_dup, max_retries=5):
    prompt = f"""
    Ты — ИИ-аудитор. Оцени предмет для лута (от 0.000 до 1.000).
    ОТВЕТ ДОЛЖЕН БЫТЬ СТРОГО JSON.
    ПРАВИЛО: Поле "score" должно содержать ТОЛЬКО ЧИСЛО (float). НИКАКИХ ФОРМУЛ, СЛОЖЕНИЙ ИЛИ ТЕКСТА В ПОЛЕ SCORE.
    Выполняй расчеты внутри, но выдавай только готовый результат.

    ДАННЫЕ:
    - Предмет: {item['name']} (Редкость: {item.get('rarity', 'common')}, Тип: {i_type})
    - Локация: {scen['loc']}
    - Группа: {scen['party']} (Уровень {scen['level']})
    - Важность: {scen['imp']:.2f}
    - Метрики: Loc={l_s}, Party={p_s}, Delta={delta}, Synergy={syn}, Is_Duplicate={is_dup}

    РУКОВОДСТВО ПО ОЦЕНКЕ:
    1. КРИТИЧЕСКИЕ ШТРАФЫ (Score 0.000-0.150):
       - Delta >= 2.
       - Synergy == 0.0.
       - Loc < 0.15 И Party < 0.15.
       - Is_Duplicate == 1.0, при условии что тип предмета НЕ 'potion' и НЕ 'scroll'. Игрокам не нужны два одинаковых магических меча!
    2. ОГРАНИЧЕНИЕ ВЕЛИКОГО ЛУТА (Artifact / Legendary):
       - Если редкость "artifact", Score может быть выше 0.250 ТОЛЬКО при Важности > 0.90.
       - Если редкость "legendary", Score может быть выше 0.350 ТОЛЬКО при Важности > 0.80.
       - Это правило перекрывает хорошие метрики! Великие вещи не лежат в обычных сундуках, даже на 20 уровне.
    3. ОГРАНИЧЕНИЕ ПРЕВЫШЕНИЯ УРОВНЯ:
       - Если Delta == 1 и Важность < 0.70 -> Score 0.050-0.150.
    4. ИДЕАЛЬНЫЙ ЛУТ (Score 0.800-1.000):
       - Все проверки пройдены, Loc>0.35, Party>0.35, Delta=0, Synergy=1.0.
    5. СИТУАТИВНЫЙ ЛУТ (Score 0.300-0.600):
       - Подходит только локации ИЛИ только партии.

    Выведи JSON: {{"reason": "краткое объяснение", "score": 0.5}}
    """

    for attempt in range(max_retries):
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system",
                     "content": "You are a data auditor. Output ONLY valid JSON. score field must be a number."},
                    {"role": "user", "content": prompt}
                ],
                model=MODEL_NAME,
                temperature=0.1,
                response_format={"type": "json_object"}
            )

            content = chat_completion.choices[0].message.content
            result = json.loads(content)

            score = float(result.get("score", 0.3))
            reason = result.get("reason", "No reason provided")

            return score, reason

        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "rate_limit" in error_msg.lower():
                time.sleep(15.0 * (attempt + 1))
            else:
                return None, f"Ошибка парсинга или API: {error_msg}. Ответ: {content if 'content' in locals() else 'None'}"

    return None, "Превышено количество попыток."


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

console.print(f"\n[bold cyan]🚀 GROQ LLM-АВТОРАЗМЕТКА ЗАПУЩЕНА (В базе уже: {total_annotated})[/bold cyan]")
console.print("Нажмите [bold red]Ctrl+C[/bold red] в любой момент, чтобы остановить процесс.\n")

target_samples = int(console.input("Сколько примеров сгенерировать за эту сессию? (Например, 5000): "))

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

        is_dup = 1.0 if random.random() < 0.05 else 0.0

        i_type = str(item.get('type', 'wondrous item')).lower()
        type_ohe = get_type_ohe(i_type)

        syn = 0.0
        for cls, allowed in CLASS_SYNERGY.items():
            if cls in found_base_classes and any(at in i_type for at in allowed):
                syn = 1.0;
                break

        score, reason = ask_llm_auditor(scen, item, l_s, p_s, delta, syn, i_type, is_dup)

        if score is None:
            progress.console.print(f"[red]⚠️ Пропуск: {reason}[/red]")
            time.sleep(3)
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

        # Пауза для Groq (у них лимит 30 запросов в минуту или 14.4k токенов в минуту на Free)
        # 2-3 секунды обычно хватает, чтобы не превышать лимит Токенов В Минуту (TPM)
        time.sleep(2.5)

console.print("\n[bold green]✅ Сессия разметки завершена![/bold green]")