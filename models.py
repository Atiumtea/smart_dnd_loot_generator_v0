# models.py
import torch.nn as nn
import numpy as np
import re

# ==========================================
# 1. КОНСТАНТЫ И СИНЕРГИЯ (МЕХАНИКА)
# ==========================================
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

# ==========================================
# 2. СЕМАНТИЧЕСКИЙ ЛОР (РАСШИРЕНИЕ ЗАПРОСОВ)
# ==========================================
CLASS_LORE = {
    "barbarian": {
        "base": "ferocious warrior, rage, unarmored defense, heavy melee weapons, raw physical power.",
        "subclasses": {
            "berserker": "frenzied strikes, immune to fear, relentless attacker.",
            "totem": "animal spirits, primal magic, bear resilience, eagle sight.",
            "zealot": "divine fury, refuses to die, religious fanatic warrior."
        }
    },
    "bard": {
        "base": "magical entertainer, inspires allies, jack of all trades, spellcaster and musician.",
        "subclasses": {
            "lore": "collector of magical secrets, cutting words, vast magical knowledge.",
            "valor": "skald, combat inspiration, medium armor and martial weapons.",
            "glamour": "fey magic, charming, majestic presence, manipulative illusions."
        }
    },
    "cleric": {
        "base": "divine magic caster, healer, holy warrior, channels the power of deities.",
        "subclasses": {
            "life": "supreme healer, heavily armored medic, radiant energy, preserves life.",
            "light": "blaster caster, holy fire, dispels darkness, burning radiance.",
            "war": "frontline battle priest, heavily armored, guides weapon strikes with divine favor."
        }
    },
    "druid": {
        "base": "nature spellcaster, shapeshifter, wild shape, commands elements and beasts.",
        "subclasses": {
            "moon": "combat shapeshifter, transforms into dangerous beasts, frontline fighter.",
            "land": "master of nature spells, recovers magic easily, bonded to specific terrains.",
            "spores": "necromantic druid, fungal magic, poison, symbiotic entities."
        }
    },
    "fighter": {
        "base": "master of martial combat, skilled with all weapons and armors, battlefield tactician.",
        "subclasses": {
            "champion": "peak physical athlete, brutal critical strikes, raw strength.",
            "battle master": "tactical commander, combat maneuvers, versatile weapon expert.",
            "cavalier": "mounted warrior, heavily armored knight, protects allies, uses lances.",
            "eldritch knight": "magical warrior, combines arcane spells with martial prowess."
        }
    },
    "monk": {
        "base": "martial artist, unarmored, uses ki energy, agile strikes, bare-handed combat.",
        "subclasses": {
            "open hand": "master of unarmed combat, trips and stuns opponents, self-healing.",
            "shadow": "ninja, teleports through darkness, stealth and infiltration.",
            "kensei": "weapon master, blends ki with swords and bows, elegant strikes."
        }
    },
    "paladin": {
        "base": "holy knight, divine smite, heavy armor, auras of protection, heals by touch.",
        "subclasses": {
            "devotion": "classic holy knight, glowing weapons, turns the unholy.",
            "vengeance": "relentless avenger, hunts down foes, aggressive damage dealer.",
            "ancients": "green knight, protects nature, wards against spell damage."
        }
    },
    "ranger": {
        "base": "wilderness survivor, tracker, uses bows and two-weapon fighting, nature magic.",
        "subclasses": {
            "hunter": "slayer of monsters, versatile combatant against hordes or giants.",
            "beast master": "fights alongside a loyal animal companion.",
            "gloom stalker": "creature of darkness, deadly first strikes, invisible in shadows."
        }
    },
    "rogue": {
        "base": "stealthy skirmisher, sneak attacks, skilled with lockpicks, avoids detection.",
        "subclasses": {
            "thief": "agile burglar, treasure hunter, climbs walls, disarms traps.",
            "assassin": "deadly killer, poison expert, strikes from the shadows, surprise attacks.",
            "arcane trickster": "magical thief, uses illusions and enchantments to steal and deceive."
        }
    },
    "sorcerer": {
        "base": "innate spellcaster, metamagic, manipulates spell effects, charismatic.",
        "subclasses": {
            "draconic": "dragon scales, elemental affinity, breathes fire or lightning.",
            "wild magic": "unpredictable chaos, surges of random magic effects.",
            "divine soul": "chosen by gods, combines cleric healing with sorcerer spells."
        }
    },
    "warlock": {
        "base": "pact magic, eldritch blast, granted power by otherworldly patrons, invocations.",
        "subclasses": {
            "fiend": "hellish power, commands fire, gains temporary health on kills.",
            "archfey": "fey tricks, charming, teleporting, illusory defenses.",
            "hexblade": "shadow magic warrior, curses foes, fights with conjured melee weapons."
        }
    },
    "wizard": {
        "base": "scholarly spellcaster, huge spellbook, rituals, master of arcane studies.",
        "subclasses": {
            "evocation": "destructive magic, fireballs, sculpts spells around allies.",
            "abjuration": "protective magic, arcane wards, counterspells.",
            "divination": "foresees the future, alters dice rolls, gathers distant intel."
        }
    },
    "artificer": {
        "base": "magical inventor, infuses items with magic, creates constructs and gadgets.",
        "subclasses": {
            "alchemist": "brews magical potions, heals allies, throws acid and fire.",
            "armorer": "wears magical power armor, acts as an impenetrable tank.",
            "battle smith": "fights with martial weapons, accompanied by a mechanical steel defender."
        }
    }
}

# ==========================================
# 3. МАССИВЫ ДЛЯ ГЕНЕРАЦИИ ПРИМЕРОВ (UI)
# ==========================================
TERRAIN = [
    "dark crypt", "abandoned mine", "city slums", "sewers network",
    "noble estate", "wizard tower", "ancient forest", "frozen tundra",
    "scorching desert", "stinking swamp", "mountain peak", "shipwreck",
    "volcanic crater", "feywild glade", "shadowfell wasteland", "astral plane",
    "underground cavern", "ruined temple", "floating island", "tavern basement"
]

PLANES = [
    "Astral Plane silver void", "Feywild enchanted forest", "Shadowfell domain of dread",
    "Nine Hells fiery battlefield", "City of Brass in the Elemental Plane of Fire",
    "Endless layers of the Abyss", "Mechanus clockwork gears", "Limbo chaotic storm"
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

# ==========================================
# 4. ВСПОМОГАТЕЛЬНАЯ ЛОГИКА
# ==========================================
def get_type_ohe(item_type_str: str) -> list:
    item_type_str = item_type_str.lower()
    ohe = [0.0] * len(ITEM_TYPES)

    for i, t in enumerate(ITEM_TYPES):
        pattern = rf'\b{t}\b'
        if re.search(pattern, item_type_str):
            ohe[i] = 1.0

    if sum(ohe) == 0:
        ohe[-1] = 1.0
    return ohe


def build_party_semantics(party_input_string: str) -> tuple[str, list[str]]:
    """
    Принимает строку вроде "Life Cleric, Assassin Rogue".
    Возвращает кортеж: (обогащенный текст для эмбеддингов, список базовых классов для проверки синергии).
    """
    party_lower = party_input_string.lower()
    enriched_parts = []
    found_base_classes = []

    for cls, lore in CLASS_LORE.items():
        if cls in party_lower:
            found_base_classes.append(cls)
            enriched_parts.append(f"{cls}: {lore['base']}")

            for sub, sub_desc in lore['subclasses'].items():
                if sub in party_lower:
                    enriched_parts.append(f"({sub} - {sub_desc})")

    final_text = " ".join(enriched_parts) if enriched_parts else party_input_string
    return final_text, found_base_classes

# ==========================================
# 5. АРХИТЕКТУРА НЕЙРОСЕТИ
# ==========================================
class DnDItemRanker(nn.Module):
    def __init__(self, input_size=None):
        super(DnDItemRanker, self).__init__()
        if input_size is None:
            input_size = 6 + len(ITEM_TYPES)

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

# ==========================================
# 6. УТИЛИТЫ ДЛЯ РЕДКОСТИ И УРОВНЕЙ (TIER BRACKETS)
# ==========================================
def get_rarity_val(rarity_str: str, party_level: int = 1) -> int:
    r = str(rarity_str).lower()
    if 'artifact' in r: return 6
    if 'legendary' in r: return 5
    if 'very rare' in r: return 4
    if 'uncommon' in r: return 2
    if re.search(r'\brare\b', r): return 3
    if 'varies' in r:
        if party_level >= 16: return 5
        elif party_level >= 10: return 4
        elif party_level >= 4: return 3
        else: return 2
    return 1

def get_tier_brackets(rarity_val: int) -> tuple[int, int]:
    mapping = {
        1: (1, 3),    # Common
        2: (1, 3),    # Uncommon
        3: (4, 9),    # Rare
        4: (10, 15),  # Very Rare
        5: (16, 20),  # Legendary
        6: (17, 20)   # Artifact
    }
    return mapping.get(rarity_val, (1, 3))


def calculate_level_delta(item_rarity_val: int, party_level: int) -> int:
    min_lvl, max_lvl = get_tier_brackets(item_rarity_val)
    delta = 0
    if party_level < min_lvl:
        delta = min_lvl - party_level
    elif party_level > max_lvl:
        delta = max_lvl - party_level

    return delta