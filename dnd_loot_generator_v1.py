import os
import logging
import warnings

os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['SAFETENSORS_FAST_GPU'] = '1'

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import random
import re
import chromadb
from sentence_transformers import SentenceTransformer, util

from models import DnDItemRanker, CLASS_SYNERGY, get_type_ohe, TERRAIN, ATMOSPHERE, ENEMY_FACTIONS, ENEMY_ACTIONS, build_party_semantics, CLASS_LORE

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

def get_expected_rarity_for_level(level):
    if level <= 4: return 2
    elif level <= 10: return 3
    elif level <= 16: return 4
    else: return 5

def roll_final_loot(valid_items, party_level):
    print("\n🎲 Бросаем виртуальные кубики...")
    if random.random() < 0.05:
        return "\n🎲 Выпала странная БЕЗДЕЛУШКА (бросьте d100 по таблице Trinkets)."

    if not valid_items:
        gold_amount = random.randint(10, 50) * party_level
        return f"\n💰 Стоящего лута нет. Вы нашли мешочек с {gold_amount} зм."

    weights = [item['final_score'] for item in valid_items]
    chosen_item = random.choices(valid_items, weights=weights, k=1)[0]
    drop_chance = (chosen_item['final_score'] / sum(weights)) * 100

    loc_s = chosen_item.get('loc_score', 0)
    party_s = chosen_item.get('party_score', 0)

    if party_s > loc_s + 0.1: reason = "Этот предмет идеально подходит способностям вашей группы."
    elif loc_s > party_s + 0.1: reason = "Этот трофей выглядит очень уместно в данной локации."
    else: reason = "Сбалансированная находка, которая вписывается в окружение и полезна героям."

    result = (
        f"\n✨ НАГРАДА: {chosen_item['name']}\n"
        f"   • Редкость: {chosen_item['rarity']}\n"
        f"   • Тип: {chosen_item['type']}\n"
        f"   • Шанс из пула: {drop_chance:.1f}%\n"
        f"   💬 Комментарий: {reason}\n"
        f"   • Описание: {str(chosen_item.get('description', ''))[:200]}..."
    )
    return result

class SmartLootGenerator:
    def __init__(self):
        print("Загрузка компонентов ИИ...")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        print("Подключение к базе знаний ChromaDB...")
        self.db_client = chromadb.PersistentClient(path="./dnd_vector_db")
        try:
            self.collection = self.db_client.get_collection(name="magic_items")
        except Exception:
            print("⚠️ Ошибка: Векторная база не найдена!")
            exit()

        try:
            with open('scaler_hybrid.pkl', 'rb') as f:
                self.current_scaler = pickle.load(f)
        except FileNotFoundError:
            print("⚠️ Ошибка: Файл 'scaler_hybrid.pkl' не найден!")
            exit()

        self.model = DnDItemRanker(input_size=15)
        self.load_model('dnd_hybrid_weights.pth')

    def load_model(self, path):
        try:
            self.model.load_state_dict(torch.load(path, weights_only=True))
            self.model.eval()
        except FileNotFoundError:
            print(f"\n⚠️ Ошибка: Файл {path} не найден!")
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
        expected_rarity = get_expected_rarity_for_level(party_level)

        for i, (doc_id, item) in enumerate(unique_candidates.items()):
            l_score = loc_scores_raw[i].item()
            p_score = party_scores_raw[i].item()

            if max(l_score, p_score) < 0.10:
                continue

            delta = get_rarity_val(item['rarity'], expected_rarity) - expected_rarity
            is_duplicate = 1.0 if str(item['name']).lower() in [inv.lower() for inv in party_inventory] else 0.0

            item_type_str = str(item.get('type', 'wondrous item')).lower()
            type_ohe_list = get_type_ohe(item_type_str)

            synergy_flag = 0.0
            for cls, allowed_types in CLASS_SYNERGY.items():
                if cls in found_base_classes:
                    if any(t in item_type_str for t in allowed_types):
                        synergy_flag = 1.0
                        break

            feature_vector = [l_score, p_score, story_importance, delta, is_duplicate, synergy_flag] + type_ohe_list
            features_list.append(feature_vector)

            item.update({
                'loc_score': l_score, 'party_score': p_score,
                'delta': delta, 'synergy': synergy_flag
            })
            candidates.append(item)

        if not candidates:
            return []

        X_raw = np.array(features_list)
        X_scaled = self.current_scaler.transform(X_raw)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        with torch.no_grad():
            predictions = self.model(X_tensor).numpy().flatten()

        for i, item in enumerate(candidates):
            item['final_score'] = float(predictions[i])

        candidates.sort(key=lambda x: x['final_score'], reverse=True)

        print("\n" + "=" * 50)
        print(" 🛠️ DEBUG: ТОП-3 ПРЕДМЕТА ГЛАЗАМИ ИИ")
        print("=" * 50)
        for i in range(min(3, len(candidates))):
            c = candidates[i]
            status = "✅ ПРОШЕЛ" if c['final_score'] >= 0.30 else "❌ ОТКЛОНЕН"
            print(f"{i + 1}. {c['name']} ({c['rarity']}) -> {status}")
            print(f"   📊 Итоговый Regression Score: {c['final_score']:.3f}")
            print(f"   ├─ Локация: {c['loc_score']:.3f}")
            print(f"   ├─ Партия:  {c['party_score']:.3f}")
            print(f"   └─ Дельта:  {c['delta']}")
            print("-" * 50)
        print("=" * 50 + "\n")

        valid_candidates = []
        for item in candidates:
            if item['final_score'] >= 0.30:
                item['final_score'] = item['final_score'] ** 3
                valid_candidates.append(item)

        valid_candidates.sort(key=lambda x: x['final_score'], reverse=True)
        return valid_candidates

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print(" 🐉 УМНЫЙ ГЕНЕРАТОР ЛУТА D&D 5e")
    print("=" * 55)

    generator = SmartLootGenerator()
    print("Система готова к работе!\n")

    while True:
        print("-" * 55)
        command = input("Нажмите [Enter] для генерации или 'q' для выхода: ").strip().lower()
        if command in ['q', 'й']:
            print("Удачных игр!")
            break

        try:
            lvl_input = input("⚔️ Уровень группы (1-20): ").strip().lower()
            if lvl_input in ['q', 'й']: break
            party_level = int(lvl_input)

            imp_input = input("🔥 Важность боя (0.0 - 1.0): ").strip().lower()
            if imp_input in ['q', 'й']: break
            story_importance = float(imp_input)

        except ValueError:
            print("⚠️ Ошибка ввода чисел.")
            continue

        dyn_loc = f"{random.choice(TERRAIN)}, {random.choice(ATMOSPHERE)}, {random.choice(ENEMY_FACTIONS)}, {random.choice(ENEMY_ACTIONS)}"
        print(f"🗺️ ЛОКАЦИЯ (Например: {dyn_loc}):")
        loc_input = input("   > ")

        party_members = []
        base_classes_list = list(CLASS_LORE.keys())
        party_size = random.randint(3, 5)
        for _ in range(party_size):
            base_cls = random.choice(base_classes_list)
            sub_cls = random.choice(list(CLASS_LORE[base_cls]['subclasses'].keys()))
            party_members.append(f"{sub_cls.capitalize()} {base_cls.capitalize()}")
        dyn_party = ", ".join(party_members)

        print(f"🛡️ СОСТАВ ПАРТИИ (Например: {dyn_party}):")
        party_input = input("   > ")

        print("\n🧠 ИИ анализирует двойной контекст...")

        pool = generator.generate_loot(
            location_text=loc_input,
            party_text=party_input,
            party_level=party_level,
            story_importance=story_importance
        )

        print(roll_final_loot(pool, party_level))
        print("-" * 55 + "\n")