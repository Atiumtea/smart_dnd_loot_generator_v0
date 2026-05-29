import os
import sys
import subprocess
import time
import logging
import warnings

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.rule import Rule

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import random
import json
import chromadb
import chromadb.errors
from sentence_transformers import SentenceTransformer, util

from models import (
    DnDItemRanker, CLASS_SYNERGY, get_type_ohe, PLANES, TERRAIN, ATMOSPHERE,
    ENEMY_FACTIONS, ENEMY_ACTIONS, build_party_semantics, CLASS_LORE,
    get_tier_brackets, get_rarity_val, calculate_level_delta
)

os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['SAFETENSORS_FAST_GPU'] = '1'
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

console = Console()

def roll_final_loot(valid_items, party_level):
    console.print("\n[bold cyan]🎲 Бросаем виртуальные кубики...[/bold cyan]")

    if random.random() < 0.01:
        console.print(Panel(
            "[bold white]Выпала странная БЕЗДЕЛУШКА[/bold white]\n[dim](бросьте d100 по таблице Trinkets в Книге Игрока).[/dim]",
            title="[bold yellow]🎲 СЛУЧАЙНОСТЬ[/bold yellow]",
            border_style="yellow",
            expand=False
        ))
        return

    if not valid_items:
        gold_amount = random.randint(10, 50) * party_level
        console.print(Panel(
            f"[bold gold1]Стоящего лута нет.[/bold gold1]\nВы нашли мешочек с {gold_amount} зм.",
            title="[bold yellow]💰 УТЕШИТЕЛЬНЫЙ ПРИЗ[/bold yellow]",
            border_style="gold1",
            expand=False
        ))
        return

    # ВЗВЕШЕННЫЙ БРОСОК
    pool_size = len(valid_items)
    weights = [item['final_score'] for item in valid_items]
    chosen_item = random.choices(valid_items, weights=weights, k=1)[0]
    drop_chance = (chosen_item['final_score'] / sum(weights)) * 100

    loc_s = chosen_item.get('loc_score', 0)
    party_s = chosen_item.get('party_score', 0)
    delta = chosen_item.get('delta', 0)
    synergy = chosen_item.get('synergy', 0)

    if party_s > loc_s + 0.1:
        reason = "Этот предмет идеально подходит способностям вашей группы."
    elif loc_s > party_s + 0.1:
        reason = "Этот трофей выглядит очень уместно в данной локации."
    else:
        reason = "Сбалансированная находка, которая вписывается в окружение и полезна героям."

    desc = str(chosen_item.get('description', '')).strip()

    content = (
        f"[bold dim]Размер пула кандидатов:[/bold dim] {pool_size} шт.\n"
        f"[bold cyan]Источник:[/bold cyan] [italic]{chosen_item.get('source', 'Unknown')}[/italic]\n"
        f"[bold cyan]Редкость:[/bold cyan] {chosen_item['rarity'].title()}\n"
        f"[bold cyan]Тип:[/bold cyan] {chosen_item['type'].title()}\n"
        f"[bold cyan]Шанс выпадения:[/bold cyan] {drop_chance:.1f}% [dim](при ML Score: {chosen_item['final_score']:.4f})[/dim]\n"
        f"[bold cyan]Loc Score (Локация):[/bold cyan] {loc_s:.4f}\n"
        f"[bold cyan]Party Score (Группа):[/bold cyan] {party_s:.4f}\n"
        f"[bold cyan]Delta (Разница уровней):[/bold cyan] {delta}\n"
        f"[bold cyan]Синергия с классами:[/bold cyan] {'Есть' if synergy > 0 else 'Нет'}\n"
        f"[bold cyan]Комментарий:[/bold cyan] [italic green]{reason}[/italic green]\n"
        f"[dim]{'─' * 65}[/dim]\n" 
        f"[bold white]Описание:[/bold white]\n{desc}"
    )

    console.print(Panel(
        content,
        title=f"[bold yellow]✨ НАГРАДА: {chosen_item['name'].upper()} ✨[/bold yellow]",
        border_style="green",
        padding=(1, 2),
        expand=True
    ))

class SmartLootGenerator:
    def __init__(self):
        with console.status("[bold green]Загрузка компонентов ИИ и векторной БД...[/bold green]", spinner="dots"):
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            self.db_client = chromadb.PersistentClient(path="./dnd_vector_db")

            try:
                self.collection = self.db_client.get_collection(name="magic_items")

                all_data = self.collection.get(include=['metadatas'])
                sources = set()
                for meta in all_data['metadatas']:
                    if meta and 'source' in meta:
                        sources.add(meta['source'])
                self.available_sources = sorted(list(sources))
                self.active_sources = self.available_sources.copy()

            except chromadb.errors.InvalidCollectionException:
                console.print(
                    "[bold red]⚠️ Ошибка: Коллекция 'magic_items' не найдена. Запустите Data Pipeline (parser.py) из главного меню![/bold red]")
                exit()

            try:
                with open('preprocessor_hybrid.pkl', 'rb') as f:
                    self.preprocessor = pickle.load(f)
            except FileNotFoundError:
                console.print(
                    "[bold red]⚠️ Ошибка: Файл 'preprocessor_hybrid.pkl' не найден. Запустите обучение модели из меню![/bold red]")
                exit()

            try:
                with open('calibration_lut.json', 'r') as f:
                    self.calibration_lut = json.load(f)
            except FileNotFoundError:
                console.print(
                    "[bold yellow]⚠️ 'calibration_lut.json' не найден. Калибровка отключена. Запустите обучение для генерации файла![/bold yellow]")
                self.calibration_lut = [0.0] * 20

            self.model = DnDItemRanker(input_size=15)
            self.load_model('dnd_hybrid_weights.pth')

        console.print("[dim]✅ ИИ-модули и база данных загружены.[/dim]")

    def calibrate_score(self, raw_score):
        idx = int(raw_score / 0.05)
        idx = min(max(0, idx), len(self.calibration_lut) - 1)
        return raw_score + self.calibration_lut[idx]

    def load_model(self, path):
        try:
            self.model.load_state_dict(torch.load(path, weights_only=True))
            self.model.eval()
        except FileNotFoundError:
            console.print(f"[bold red]\n⚠️ Ошибка: Файл {path} не найден![/bold red]")
            exit()

    def configure_sources(self):
        console.print("\n[bold yellow]📚 НАСТРОЙКА ИСТОЧНИКОВ ЛУТА[/bold yellow]")

        table = Table(box=box.MINIMAL_DOUBLE_HEAD)
        table.add_column("ID", justify="right", style="cyan")
        table.add_column("Название книги / Источника", style="white")

        for i, src in enumerate(self.available_sources, 1):
            table.add_row(str(i), src)

        console.print(table)

        console.print(
            "[dim]Введите номера источников через запятую, которые хотите ОСТАВИТЬ (например: 1, 3, 5).[/dim]")
        console.print("[dim]Или введите [bold]all[/bold], чтобы использовать все доступные книги.[/dim]")

        while True:
            choice = console.input("[bold]Ваш выбор (по умолчанию 'all'): [/bold]").strip().lower()
            if not choice or choice == 'all':
                self.active_sources = self.available_sources.copy()
                console.print(f"[green]✅ Все источники ({len(self.active_sources)} шт.) включены.[/green]\n")
                break

            try:
                selected_indices = [int(x.strip()) - 1 for x in choice.split(',')]
                valid_sources = []
                for idx in selected_indices:
                    if 0 <= idx < len(self.available_sources):
                        valid_sources.append(self.available_sources[idx])

                if valid_sources:
                    self.active_sources = valid_sources
                    console.print(f"[green]✅ Выбрано источников: {len(self.active_sources)}.[/green]\n")
                    break
                else:
                    console.print("[red]❌ Некорректный ввод. Укажите существующие номера.[/red]")
            except ValueError:
                console.print("[red]❌ Некорректный формат. Используйте числа через запятую.[/red]")

    def generate_loot(self, location_text, party_text, party_level, story_importance, party_inventory=[]):
        if not self.active_sources:
            return []

        semantic_party, found_base_classes = build_party_semantics(party_text)

        with torch.no_grad():
            loc_emb = self.encoder.encode(location_text)
            party_emb = self.encoder.encode(semantic_party)

        where_clause = None
        if len(self.active_sources) < len(self.available_sources):
            if len(self.active_sources) == 1:
                where_clause = {"source": self.active_sources[0]}
            else:
                where_clause = {"source": {"$in": self.active_sources}}

        results = self.collection.query(
            query_embeddings=[loc_emb.tolist(), party_emb.tolist()],
            n_results=400,
            where=where_clause,
            include=['metadatas', 'documents', 'embeddings']
        )

        unique_candidates = {}
        for q_idx in range(2):
            if not results['ids'][q_idx]:
                continue
            for i, doc_id in enumerate(results['ids'][q_idx]):
                if doc_id not in unique_candidates:
                    unique_candidates[doc_id] = {
                        'name': results['metadatas'][q_idx][i]['name'],
                        'type': results['metadatas'][q_idx][i]['type'],
                        'rarity': results['metadatas'][q_idx][i]['rarity'],
                        'source': results['metadatas'][q_idx][i].get('source', 'Unknown'),
                        'description': results['documents'][q_idx][i],
                        'embedding': results['embeddings'][q_idx][i]
                    }

        if not unique_candidates:
            return []

        candidates_embs = torch.tensor([c['embedding'] for c in unique_candidates.values()], dtype=torch.float32)
        loc_emb_tensor = torch.tensor(loc_emb, dtype=torch.float32)
        party_emb_tensor = torch.tensor(party_emb, dtype=torch.float32)

        loc_scores_raw = util.cos_sim(loc_emb_tensor, candidates_embs)[0]
        party_scores_raw = util.cos_sim(party_emb_tensor, candidates_embs)[0]

        features_list = []
        candidates = []

        for i, (doc_id, item) in enumerate(unique_candidates.items()):
            l_score = loc_scores_raw[i].item()
            p_score = party_scores_raw[i].item()

            if max(l_score, p_score) < 0.15:
                continue

            rarity_val = get_rarity_val(item['rarity'], party_level)
            delta = calculate_level_delta(rarity_val, party_level)

            is_duplicate = 1.0 if str(item['name']).lower() in [inv.lower() for inv in party_inventory] else 0.0

            item_type_str = str(item.get('type', 'wondrous item')).lower()
            type_ohe_list = get_type_ohe(item_type_str)

            synergy_flag = 0.0
            for cls, allowed_types in CLASS_SYNERGY.items():
                if cls in found_base_classes:
                    if any(t in item_type_str for t in allowed_types):
                        synergy_flag = 1.0
                        break

            continuous_features = [l_score, p_score, story_importance, delta]
            binary_features = [is_duplicate, synergy_flag]
            feature_vector = continuous_features + binary_features + type_ohe_list

            features_list.append(feature_vector)

            item.update({
                'loc_score': l_score, 'party_score': p_score,
                'delta': delta, 'synergy': synergy_flag
            })
            candidates.append(item)

        if not candidates:
            return []

        X_raw = np.array(features_list, dtype=np.float32)
        X_scaled = self.preprocessor.transform(X_raw)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        with torch.no_grad():
            predictions = self.model(X_tensor).numpy().flatten()

        for i, item in enumerate(candidates):
            raw_score = float(predictions[i])
            calibrated_score = self.calibrate_score(raw_score)
            item['final_score'] = min(1.0, max(0.0, calibrated_score))

        candidates.sort(key=lambda x: x['final_score'], reverse=True)

        base_score_threshold = 0.40
        valid_candidates = [c for c in candidates if c['final_score'] >= base_score_threshold]

        if not valid_candidates:
            base_score_threshold = 0.20
            valid_candidates = [c for c in candidates if c['final_score'] >= base_score_threshold]

        console.print()
        table = Table(title="[dim]🛠️ DEBUG: ТОП-3 ПРЕДМЕТА ГЛАЗАМИ модели[/dim]", box=box.SIMPLE)
        table.add_column("Название", style="cyan")
        table.add_column("Редкость", style="magenta")
        table.add_column("Источник", style="yellow")
        table.add_column("Скор (L | P | D)", justify="right", style="white")
        table.add_column("Статус", justify="center")

        for i in range(min(3, len(candidates))):
            c = candidates[i]
            status = "[bold green]✅ В ПУЛЕ[/bold green]" if c[
                                                                'final_score'] >= base_score_threshold else "[bold red]❌ ОТКЛОНЕН[/bold red]"
            score_str = f"{c['final_score']:.3f} ([dim]{c['loc_score']:.2f} | {c['party_score']:.2f} | {c['delta']}[/dim])"

            source_short = c['source'][:15] + "..." if len(c['source']) > 15 else c['source']
            table.add_row(c['name'], c['rarity'].title(), source_short, score_str, status)

        console.print(table)

        return valid_candidates

def run_generator():
    os.system('cls' if os.name == 'nt' else 'clear')
    console.print(Rule(title="[bold green]🐉 УМНЫЙ ГЕНЕРАТОР ЛУТА D&D 5e 🐉[/bold green]", style="green"))
    console.print()

    try:
        generator = SmartLootGenerator()
        generator.configure_sources()
    except Exception as e:
        console.print(f"\n[bold red]Ошибка инициализации: {e}[/bold red]")
        return

    while True:
        console.print(Rule(style="dim"))
        command = console.input(
            "[bold white]Нажмите [Enter] для генерации или 'q' для выхода в меню:[/bold white] ").strip().lower()
        if command in ['q', 'й']:
            console.print("[italic green]Возврат в главное меню...[/italic green] 🎲")
            break

        try:
            dyn_lvl = random.randint(1, 20)
            console.print(f"[bold cyan]⚔️  Уровень группы (1-20)[/bold cyan] [dim](Например: {dyn_lvl}):[/dim]")
            lvl_input = console.input("   [bold]>[/bold] ").strip().lower()
            if lvl_input in ['q', 'й']: break
            party_level = int(lvl_input) if lvl_input else dyn_lvl

            dyn_imp = round(random.uniform(0.1, 1.0), 2)
            console.print(f"[bold red]🔥 Важность боя (0.0 - 1.0)[/bold red] [dim](Например: {dyn_imp}):[/dim]")
            imp_input = console.input("   [bold]>[/bold] ").strip().lower()
            if imp_input in ['q', 'й']: break
            story_importance = float(imp_input) if imp_input else dyn_imp

        except ValueError:
            console.print("[bold red]⚠️ Ошибка: Вводите только числа![/bold red]")
            continue

        terrain_str = f"{random.choice(TERRAIN)}, {random.choice(PLANES)}" if random.random() < 0.2 else random.choice(
            TERRAIN)
        dyn_loc = f"{terrain_str}, {random.choice(ATMOSPHERE)}, {random.choice(ENEMY_FACTIONS)}, {random.choice(ENEMY_ACTIONS)}"
        console.print(f"[bold yellow]🗺️  ЛОКАЦИЯ[/bold yellow] [dim](Например: {dyn_loc}):[/dim]")
        loc_input = console.input("   [bold]>[/bold] ")
        if not loc_input.strip():
            loc_input = dyn_loc

        party_members = []
        base_classes_list = list(CLASS_LORE.keys())
        party_size = random.randint(3, 5)
        for _ in range(party_size):
            base_cls = random.choice(base_classes_list)
            sub_cls = random.choice(list(CLASS_LORE[base_cls]['subclasses'].keys()))
            party_members.append(f"{sub_cls.capitalize()} {base_cls.capitalize()}")
        dyn_party = ", ".join(party_members)

        console.print(f"[bold yellow]🛡️  СОСТАВ ПАРТИИ[/bold yellow] [dim](Например: {dyn_party}):[/dim]")
        party_input = console.input("   [bold]>[/bold] ")
        if not party_input.strip():
            party_input = dyn_party

        with console.status("[bold purple]🧠 ИИ анализирует двойной контекст...[/bold purple]", spinner="bouncingBar"):
            pool = generator.generate_loot(
                location_text=loc_input,
                party_text=party_input,
                party_level=party_level,
                story_importance=story_importance
            )

        roll_final_loot(pool, party_level)

def main_menu():
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        console.print(Rule(title="[bold green]⚙️ ПУЛЬТ УПРАВЛЕНИЯ ML-ПАЙПЛАЙНОМ ⚙️[/bold green]", style="green"))

        table = Table(box=box.MINIMAL_DOUBLE_HEAD)
        table.add_column("Команда", justify="center", style="cyan")
        table.add_column("Модуль", style="bold white")
        table.add_column("Описание процесса", style="dim")

        table.add_row("1", "Data Pipeline (parser.py)", "Сбор данных по API, векторизация и обновление ChromaDB")
        table.add_row("2", "Data Annotation (llm_annotator.py)", "Генерация датасета через Gatekeeper эвристики и LLM")
        table.add_row("3", "Model Training (train_hybrid_evaluate.py)",
                      "Обучение нейросети, расчет метрик и обновление LUT-калибровки")
        table.add_row("4", "Loot Generator (Inference)", "Запуск финального боевого генератора наград")
        table.add_row("q", "Выход", "Закрыть пульт управления")

        console.print(table)

        choice = console.input("\n[bold yellow]Выберите команду:[/bold yellow] ").strip().lower()

        if choice == '1':
            console.print("\n[bold cyan]🚀 Запуск parser.py...[/bold cyan]")
            subprocess.run([sys.executable, "parser.py"])
            console.input("\n[dim]Нажмите Enter для возврата в меню...[/dim]")

        elif choice == '2':
            console.print("\n[bold cyan]🚀 Запуск llm_annotator.py...[/bold cyan]")
            subprocess.run([sys.executable, "llm_annotator.py"])
            console.input("\n[dim]Нажмите Enter для возврата в меню...[/dim]")

        elif choice == '3':
            console.print("\n[bold cyan]🚀 Запуск train_hybrid_evaluate.py...[/bold cyan]")
            subprocess.run([sys.executable, "train_hybrid_evaluate.py"])
            console.input("\n[dim]Нажмите Enter для возврата в меню...[/dim]")

        elif choice == '4':
            run_generator()

        elif choice in ['q', 'й']:
            console.print("[italic green]Завершение работы. Удачных игр![/italic green] 👋")
            break

        else:
            console.print("[red]❌ Неизвестная команда. Выберите пункт от 1 до 4 или 'q' для выхода.[/red]")
            time.sleep(1.5)

if __name__ == "__main__":
    main_menu()