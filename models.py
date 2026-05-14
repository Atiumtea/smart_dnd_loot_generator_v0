import torch.nn as nn
import numpy as np
import re

# === МАССИВЫ ДЛЯ ГЕНЕРАЦИИ ПРИМЕРОВ ЛОКАЦИЙ ===
TERRAIN = [
    "dark crypt", "abandoned mine", "city slums", "sewers network",
    "noble estate", "wizard tower", "ancient forest", "frozen tundra",
    "scorching desert", "stinking swamp", "mountain peak", "shipwreck",
    "volcanic crater", "feywild glade", "shadowfell wasteland", "astral plane",
    "underground cavern", "ruined temple", "floating island", "tavern basement"
]

ATMOSPHERE = [
    "thick fog", "heavy rain", "pitch black", "cobwebs and dust",
    "smell of sulfur", "glowing arcane runes", "howling blizzard",
    "eerie silence", "bloodstains", "magical twilight", "overgrown with vines",
    "crumbling walls", "knee-deep mud", "oppressive heat", "toxic fumes"
]

ENEMY_FACTIONS = [
    "bandit highwaymen", "pirate mutineers", "doomsday cultists", "drow assassins",
    "undead horde", "vampire spawn", "necromancer and skeletons",
    "mind flayer colony", "beholder", "yuan-ti abominations", "giant spiders",
    "mimics and ropers", "young red dragon", "hag coven", "fire elementals",
    "abyssal demons", "goblin raiding party", "rogue artificer constructs"
]

ENEMY_ACTIONS = [
    "setting up an ambush", "guarding a locked chest", "conducting a dark ritual",
    "sleeping", "patrolling the area", "fighting a rival group",
    "interrogating a prisoner", "feasting on a corpse", "searching for intruders",
    "repairing their weapons", "hiding in the shadows", "worshipping an idol"
]

ITEM_TYPES = [
    'weapon', 'armor', 'potion', 'ring', 'scroll',
    'wand', 'staff', 'rod', 'wondrous item'
]

CLASS_SYNERGY = {
    'barbarian': ['weapon', 'potion', 'ring', 'wondrous item'],
    'monk': ['weapon', 'potion', 'ring', 'wondrous item'],
    'fighter': ['weapon', 'armor', 'potion', 'ring', 'wondrous item'],
    'rogue': ['weapon', 'armor', 'potion', 'ring', 'wondrous item'],
    'paladin': ['weapon', 'armor', 'potion', 'ring', 'scroll', 'wondrous item'],
    'ranger': ['weapon', 'armor', 'potion', 'ring', 'scroll', 'wondrous item'],
    'cleric': ['weapon', 'armor', 'potion', 'ring', 'scroll', 'rod', 'staff', 'wondrous item'],
    'druid': ['weapon', 'armor', 'potion', 'ring', 'scroll', 'staff', 'wondrous item'],
    'bard': ['weapon', 'armor', 'potion', 'ring', 'scroll', 'wand', 'staff', 'wondrous item'],
    'wizard': ['potion', 'ring', 'scroll', 'wand', 'staff', 'rod', 'wondrous item'],
    'sorcerer': ['potion', 'ring', 'scroll', 'wand', 'staff', 'rod', 'wondrous item'],
    'warlock': ['weapon', 'potion', 'ring', 'scroll', 'wand', 'staff', 'rod', 'wondrous item'],
    'artificer': ['weapon', 'armor', 'potion', 'ring', 'scroll', 'wand', 'staff', 'rod', 'wondrous item']
}


def get_type_ohe(item_type_str: str) -> list:
    item_type_str = item_type_str.lower()
    ohe = [0.0] * len(ITEM_TYPES)

    for i, t in enumerate(ITEM_TYPES):
        pattern = rf'\b{t}\b'
        if re.search(pattern, item_type_str):
            ohe[i] = 1.0
            break

    if sum(ohe) == 0:
        ohe[-1] = 1.0
    return ohe

class DnDItemRanker(nn.Module):
    def __init__(self, input_size=15):
        super(DnDItemRanker, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)