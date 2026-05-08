# models.py
import torch.nn as nn
import numpy as np

# Вместо словаря с числами (0.1, 0.2), используем список для One-Hot Encoding
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
    """
    Превращает строковый тип предмета в One-Hot Encoding массив из 9 элементов.
    Например, 'armor' -> [0, 1, 0, 0, 0, 0, 0, 0, 0]
    """
    item_type_str = item_type_str.lower()
    ohe = [0.0] * len(ITEM_TYPES)
    for i, t in enumerate(ITEM_TYPES):
        if t in item_type_str:
            ohe[i] = 1.0
            break
    # Если тип не найден, по умолчанию делаем его 'wondrous item' (последний индекс)
    if sum(ohe) == 0:
        ohe[-1] = 1.0
    return ohe

# Теперь input_size = 15 (6 базовых признаков + 9 признаков типа)
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