import os
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
import re
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

    if random.random() < 0.05:
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

    if party_s > loc_s + 0.1: reason = "Этот предмет идеально подходит способностям вашей группы."
    elif loc_s > party_s + 0.1: reason = "Этот трофей выглядит очень уместно в данной локации."
    else: reason = "Сбалансированная находка, которая вписывается в окружение и полезна героям."

    desc = str(chosen_item.get('description', '')).strip()

    content = (
        f"[bold dim]Размер пула кандидатов:[/bold dim] {pool_size} шт.\n"
        f"[bold cyan]Редкость:[/bold cyan] {chosen_item['rarity'].title()}\n"
        f"[bold cyan]Тип:[/bold cyan] {chosen_item['type'].title()}\n"
        f"[bold cyan]Шанс выпадения:[/bold cyan] {drop_chance:.1f}% [dim](при ML Score: {chosen_item['final_score']:.4f})[/dim]\n"
        f"[bold cyan]Комментарий ИИ:[/bold cyan] [italic green]{reason}[/italic green]\n"
        f"{'-' * 40}\n"
        f"[bold white]Описание:[/bold white]\n{desc}"
    )

    console.print(Panel(
        content,
        title=f"[bold yellow]✨ НАГРАДА: {chosen_item['name'].upper()} ✨[/bold yellow]",
        border_style="green",
        padding=(1, 2)
    ))


class SmartLootGenerator:
    def __init__(self):
        with console.status("[bold green]Загрузка компонентов ИИ...[/bold green]", spinner="dots"):
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            self.db_client = chromadb.PersistentClient(path="./dnd_vector_db")

            try:
                self.collection = self.db_client.get_collection(name="magic_items")
            except chromadb.errors.InvalidCollectionException:
                console.print(
                    "[bold red]⚠️ Ошибка: Коллекция 'magic_items' не найдена в векторной базе.[/bold red]")
                exit()

            try:
                with open('preprocessor_hybrid.pkl', 'rb') as f:
                    self.preprocessor = pickle.load(f)
            except FileNotFoundError:
                console.print("[bold red]⚠️ Ошибка: Файл 'preprocessor_hybrid.pkl' не найден.[/bold red]")
                exit()

            self.model = DnDItemRanker(input_size=15)
            self.load_model('dnd_hybrid_weights.pth')

        console.print("[dim]✅ ИИ-модули и база данных загружены.[/dim]")

    def load_model(self, path):
        try:
            self.model.load_state_dict(torch.load(path, weights_only=True))
            self.model.eval()
        except FileNotFoundError:
            console.print(f"[bold red]\n⚠️ Ошибка: Файл {path} не найден![/bold red]")
            exit()

    def generate_loot(self, location_text, party_text, party_level, story_importance, party_inventory=[]):
        semantic_party, found_base_classes = build_party_semantics(party_text)

        with torch.no_grad():
            loc_emb = self.encoder.encode(location_text)
            party_emb = self.encoder.encode(semantic_party)

        results = self.collection.query(
            query_embeddings=[loc_emb.tolist(), party_emb.tolist()],
            n_results=400,
            include=['metadatas', 'documents', 'embeddings']
        )

        unique_candidates = {}
        for q_idx in range(2):
            for i, doc_id in enumerate(results['ids'][q_idx]):
                if doc_id not in unique_candidates:
                    unique_candidates[doc_id] = {
                        'name': results['metadatas'][q_idx][i]['name'],
                        'type': results['metadatas'][q_idx][i]['type'],
                        'rarity': results['metadatas'][q_idx][i]['rarity'],
                        'description': results['documents'][q_idx][i],
                        'embedding': results['embeddings'][q_idx][i]
                    }

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

            # СТРОГИЙ ПОРЯДОК: [Непрерывные] + [Бинарные] + [OHE Типы]
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

        # 1. Трансформация правильным препроцессором
        X_raw = np.array(features_list, dtype=np.float32)
        X_scaled = self.preprocessor.transform(X_raw)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        # 2. Инференс
        with torch.no_grad():
            predictions = self.model(X_tensor).numpy().flatten()

        for i, item in enumerate(candidates):
            item['final_score'] = float(predictions[i])

        candidates.sort(key=lambda x: x['final_score'], reverse=True)

        # 3. Динамический Базовый Скор
        base_score_threshold = 0.40
        valid_candidates = [c for c in candidates if c['final_score'] >= base_score_threshold]

        if not valid_candidates:
            base_score_threshold = 0.20
            valid_candidates = [c for c in candidates if c['final_score'] >= base_score_threshold]

        console.print()
        table = Table(title="[dim]🛠️ DEBUG: ТОП-3 ПРЕДМЕТА ГЛАЗАМИ ИИ[/dim]", box=box.SIMPLE)
        table.add_column("Название", style="cyan")
        table.add_column("Редкость", style="magenta")
        table.add_column("Скор (L | P | D)", justify="right", style="white")
        table.add_column("Статус", justify="center")

        for i in range(min(3, len(candidates))):
            c = candidates[i]
            status = "[bold green]✅ В ПУЛЕ[/bold green]" if c['final_score'] >= base_score_threshold else "[bold red]❌ ОТКЛОНЕН[/bold red]"
            score_str = f"{c['final_score']:.3f} ([dim]{c['loc_score']:.2f} | {c['party_score']:.2f} | {c['delta']}[/dim])"
            table.add_row(c['name'], c['rarity'].title(), score_str, status)

        console.print(table)

        return valid_candidates


if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    console.print(Rule(title="[bold green]🐉 УМНЫЙ ГЕНЕРАТОР ЛУТА D&D 5e 🐉[/bold green]", style="green"))
    console.print()

    generator = SmartLootGenerator()

    while True:
        console.print(Rule(style="dim"))
        command = console.input(
            "[bold white]Нажмите [Enter] для генерации или 'q' для выхода:[/bold white] ").strip().lower()
        if command in ['q', 'й']:
            console.print("[italic green]Удачных игр![/italic green] 🎲")
            break

        try:
            lvl_input = console.input("[bold cyan]⚔️  Уровень группы (1-20):[/bold cyan] ").strip().lower()
            if lvl_input in ['q', 'й']: break
            party_level = int(lvl_input)

            imp_input = console.input("[bold red]🔥 Важность боя (0.0 - 1.0):[/bold red] ").strip().lower()
            if imp_input in ['q', 'й']: break
            story_importance = float(imp_input)

        except ValueError:
            console.print("[bold red]⚠️ Ошибка: Вводите только числа![/bold red]")
            continue

        terrain_str = f"{random.choice(TERRAIN)}, {random.choice(PLANES)}" if random.random() < 0.2 else random.choice(TERRAIN)
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