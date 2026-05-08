# models.py
import torch.nn as nn
import numpy as np
import re

# Список типов оставляем без изменений
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
    Превращает строковый тип предмета в One-Hot Encoding массив.
    ИСХОДНИК ИСПРАВЛЕН: Теперь используется строгое совпадение по словам (regex).
    """
    item_type_str = item_type_str.lower()
    ohe = [0.0] * len(ITEM_TYPES)

    for i, t in enumerate(ITEM_TYPES):
        # Ищем точное вхождение слова как самостоятельного элемента, а не куска другого слова
        pattern = rf'\b{t}\b'
        if re.search(pattern, item_type_str):
            ohe[i] = 1.0
            break

    # Если тип не найден, по умолчанию делаем его 'wondrous item' (последний индекс)
    if sum(ohe) == 0:
        ohe[-1] = 1.0
    return ohe


# input_size = 15 (6 базовых признаков + 9 признаков типа)
class DnDItemRanker(nn.Module):
    def __init__(self, input_size=15):
        super(DnDItemRanker, self).__init__()
        # ИСХОДНИК ИСПРАВЛЕН: Увеличена емкость сети (Capacity).
        # Добавлен еще один слой, расширено количество нейронов и Dropout.
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)