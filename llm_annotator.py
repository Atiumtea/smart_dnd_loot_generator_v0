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


def ask_llm_auditor(scen, item, l_s, p_s, delta, syn, i_type, is_dup, max_retries=5):
    rarity_clean = str(item.get('rarity', 'common')).lower()

    # --- 1. ПРЕПРОЦЕССИНГ (Перевод в естественный язык для LLM) ---
    is_dup_text = "ДА" if is_dup == 1.0 else "НЕТ"
    syn_text = "ЕСТЬ (Предмет отлично подходит классам)" if syn == 1.0 else "НЕТ (Никто не сможет эффективно использовать)"
    consumable_text = "ДА" if any(c in i_type for c in ['potion', 'scroll']) else "НЕТ"

    if delta == 0:
        delta_text = "Идеально соответствует уровню группы."
    elif delta > 0:
        delta_text = f"На {delta} тира ВЫШЕ нормы (Предмет сильнее, чем положено)."
    else:
        delta_text = f"На {abs(delta)} тира НИЖЕ нормы (Слабый предмет)."

    # --- 2. СИСТЕМНЫЙ ПРОМПТ (Абсолютные правила) ---
    system_prompt = """Ты — строгий Dungeon Master и Data-Аудитор датасета для D&D 5e.
Твоя задача — оценить полезность лута (score от 0.000 до 1.000).
ОТВЕТ СТРОГО В ФОРМАТЕ JSON с полями "reasoning" (твоя логика) и "score" (число). Сначала пиши reasoning!

ОБЯЗАТЕЛЬНЫЕ ПРАВИЛА (ШТРАФЫ И ЗАПРЕТЫ - Приводят к score от 0.0 до 0.15):
1. ЛЕГЕНДАРКИ И АРТЕФАКТЫ: 
   - Если редкость "artifact", он МОЖЕТ выпасть ТОЛЬКО если Важность события >= 0.90.
   - Если редкость "legendary", он МОЖЕТ выпасть ТОЛЬКО если Важность события >= 0.80.
   - ВНИМАНИЕ: Если Важность события НИЖЕ этих порогов, score ВСЕГДА 0.0, даже если группа 20 уровня! Артефакты из случайных сундуков ломают лор.
2. ДУБЛИКАТЫ: Если у партии уже есть этот предмет (Дубликат = ДА) и это НЕ расходник, то предмет им не нужен (score <= 0.1).
3. СЛОМАННЫЙ БАЛАНС: Если предмет на 2 или более тира ВЫШЕ нормы (Delta >= 2), он сломает баланс. (score <= 0.15).
4. БЕСПОЛЕЗНОСТЬ: Если Синергии НЕТ, никто не сможет использовать предмет. (score <= 0.15).

ПРАВИЛА ОЦЕНКИ (Если штрафов нет):
- Идеальный лут (0.800 - 1.000): Высокое совпадение с Локацией (> 0.3) и Партией (> 0.3), идеальный баланс (Delta 0 или 1).
- Хороший лут (0.500 - 0.799): Балансный предмет, но может не идеально подходить под лор или билды.
- Слабый/Ситуативный (0.200 - 0.499): Предмет слабее уровня партии (Delta < 0) или плохо сочетается с окружением.
"""

    # --- 3. ПОЛЬЗОВАТЕЛЬСКИЙ ПРОМПТ (Чистые данные) ---
    user_prompt = f"""ДАННЫЕ О СИТУАЦИИ:
- Предмет: {item['name']}
- Тип: {i_type}
- Редкость: {rarity_clean}
- Уровень группы: {scen['level']}
- Важность события (0.0 - 1.0): {scen['imp']:.2f}
- У партии уже есть этот предмет: {is_dup_text} (Является ли расходником: {consumable_text})
- Баланс уровня (Delta): {delta_text}
- Синергия с группой: {syn_text}
- Уместность в Локации (0.0 - 1.0): {l_s:.2f}
- Полезность для Партии (0.0 - 1.0): {p_s:.2f}

Проведи оценку строго по правилам. Выдай JSON."""

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

            # Страховочный парсинг JSON (иногда модели оборачивают json в markdown)
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                match = re.search(r'\{.*\}', content, re.DOTALL)
                result = json.loads(match.group(0)) if match else {"score": 0.3, "reasoning": "Parse error"}

            score = float(result.get("score", 0.3))
            reason = result.get("reasoning", "No reason provided")

            return score, reason

        except Exception as e:
            error_msg = str(e).lower()
            if any(code in error_msg for code in ["429", "rate_limit", "timeout", "503", "502"]):
                wait_time = 15.0 * (attempt + 1)
                console.print(
                    f"[yellow]⏳ Задержка сервера (Таймаут/Лимит). Ждем {wait_time} сек... (Попытка {attempt + 1}/{max_retries})[/yellow]")
                time.sleep(wait_time)
            else:
                return None, f"Ошибка API/Парсинга: {str(e)}"

    return None, "Превышено количество попыток достучаться до сервера Groq."


# --- ЗАГРУЗКА И ОСНОВНОЙ ЦИКЛ ---
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

        # Запрашиваем LLM
        score, reason = ask_llm_auditor(scen, item, l_s, p_s, delta, syn, i_type, is_dup)

        if score is None:
            progress.console.print(f"[red]⚠️ Пропуск: {reason}[/red]")
            time.sleep(3)
            continue

        # --- 4. СОЗДАНИЕ НЕПРЕРЫВНОЙ ДИСПЕРСИИ ДЛЯ ML (Генерация Шума) ---
        # Если модель отсекла предмет (score около 0), штраф оставляем сильным.
        # Если модель дала оценку, добавляем реалистичный "шум".
        if score > 0.15:
            noise = random.gauss(0, 0.03)  # Сдвиг на +/- 3%
        elif score > 0.0:
            noise = random.gauss(0, 0.01)  # Сдвиг на +/- 1%
        else:
            noise = 0.0  # Оставляем жесткий 0.0 для фатальных ошибок

        target_y = max(0.0, min(1.0, round(score + noise, 4)))

        row_dict = {
            'item_name': item['name'], 'location_text': scen['loc'], 'party_text': scen['party'],
            'loc_score': l_s, 'party_score': p_s, 'story_importance': scen['imp'],
            'level_rarity_delta': delta, 'is_duplicate': is_dup, 'synergy_flag': syn, 'target_y': target_y
        }
        for i, t in enumerate(ITEM_TYPES):
            row_dict[f'type_{t.replace(" ", "_")}'] = type_ohe[i]

        pd.DataFrame([row_dict]).to_csv(GOLD_FILE, mode='a', header=False, index=False, sep=';')

        # Ограничиваем вывод reason в консоли до 120 символов, чтобы не спамить
        short_reason = reason if len(reason) < 120 else reason[:117] + "..."
        progress.console.print(
            f"[dim]Лут:[/dim] [cyan]{item['name'][:25]:<25}[/cyan] | "
            f"[white]Y: {target_y:.3f}[/white] | "
            f"[italic green]{short_reason}[/italic green]"
        )

        success_count += 1
        progress.update(task, advance=1)
        time.sleep(2.5)

console.print("\n[bold green]✅ Сессия разметки завершена![/bold green]")