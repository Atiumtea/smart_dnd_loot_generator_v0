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
    'ranger': ['weapon', 'armor', 'potion', 'ring', 'scroll', 'wondrous item', 'arrow'],
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
            "ancestral guardian": "summons spectral ancestors, protects allies, redirects damage.",
            "battlerager": "dwarven warrior, spiked armor, aggressive grappling, throws caution to the wind.",
            "beast": "grows natural weapons, feral mutations, claws and bites in a rage.",
            "berserker": "frenzied strikes, immune to fear, relentless attacker.",
            "giant": "grows in size, massive reach, hurls enemies and weapons, elemental strikes.",
            "storm herald": "radiates elemental auras, controls desert fire, sea lightning, or tundra ice.",
            "totem warrior": "animal spirits, primal magic, bear resilience, eagle sight.",
            "wild magic": "unpredictable magical surges during rage, bolsters allies.",
            "zealot": "divine fury, refuses to die, religious fanatic warrior."
        }
    },
    "bard": {
        "base": "magical entertainer, inspires allies, jack of all trades, spellcaster and musician.",
        "subclasses": {
            "creation": "animates objects, creates items from nothing, song of creation.",
            "eloquence": "master orator, silver-tongued, manipulates social encounters, reliable speech.",
            "glamour": "fey magic, charming, majestic presence, manipulative illusions.",
            "lore": "collector of magical secrets, cutting words, vast magical knowledge.",
            "spirits": "tells ghostly tales, channels spirits, seances, random mystical effects.",
            "swords": "blade dancer, weapon flourishes, mobile melee combatant.",
            "valor": "skald, combat inspiration, medium armor and martial weapons.",
            "whispers": "psychic blades, steals shadows, manipulative spies, sows fear."
        }
    },
    "cleric": {
        "base": "divine magic caster, healer, holy warrior, channels the power of deities.",
        "subclasses": {
            "arcana": "mystic magic, spellbreaker, combines wizard spells with divine power.",
            "death": "necromancy, reaper of souls, martial weapons, necrotic damage.",
            "forge": "master smith, heavily armored, imbues magic into weapons and armor.",
            "grave": "preserves the line between life and death, denies critical hits, detects undead.",
            "knowledge": "mind reader, vast intellect, skilled in many languages and tools.",
            "life": "supreme healer, heavily armored medic, radiant energy, preserves life.",
            "light": "blaster caster, holy fire, dispels darkness, burning radiance.",
            "nature": "druidic magic, commands animals and plants, heavy armor in the wilderness.",
            "order": "enforces law, commands allies to strike, mind control, heavily armored.",
            "peace": "bonds allies together, non-violent resolutions, prevents damage.",
            "tempest": "storm lord, wrath of nature, lightning and thunder, heavy armor.",
            "trickery": "illusionist, stealthy priest, invokes duplicates, charming and deceptive.",
            "twilight": "guardian of the night, grants darkvision, temporary hit points, heavily armored.",
            "war": "frontline battle priest, heavily armored, guides weapon strikes with divine favor."
        }
    },
    "druid": {
        "base": "nature spellcaster, shapeshifter, wild shape, commands elements and beasts.",
        "subclasses": {
            "dreams": "feywild healer, safe resting, teleports allies, soothing magic.",
            "land": "master of nature spells, recovers magic easily, bonded to specific terrains.",
            "moon": "combat shapeshifter, transforms into dangerous beasts, frontline fighter.",
            "shepherd": "summoner of beasts and fey, animal spirit totems, speaks with animals.",
            "spores": "necromantic druid, fungal magic, poison, symbiotic entities.",
            "stars": "cosmic magic, starry forms, reads omens, shoots guiding bolts.",
            "wildfire": "controls roaring flames, accompanied by a fiery spirit companion, destructive healing."
        }
    },
    "fighter": {
        "base": "master of martial combat, skilled with all weapons and armors, battlefield tactician.",
        "subclasses": {
            "arcane archer": "magical arrows, elemental shots, precise ranged combat.",
            "battle master": "tactical commander, combat maneuvers, versatile weapon expert.",
            "cavalier": "mounted warrior, heavily armored knight, protects allies, defensive tactics.",
            "champion": "peak physical athlete, brutal critical strikes, raw strength.",
            "echo knight": "time magic, summons a shadowy clone, attacks from multiple positions.",
            "eldritch knight": "magical warrior, combines arcane spells with martial prowess.",
            "psi warrior": "telekinetic strikes, psionic shields, moves with mind power.",
            "purple dragon knight": "inspiring leader, heals allies through sheer resolve, battlefield envoy.",
            "rune knight": "giant magic, carves magical runes into gear, grows to large size.",
            "samurai": "unbreakable resolve, rapid precise strikes, fighting spirit, elegant warrior."
        }
    },
    "monk": {
        "base": "martial artist, unarmored, uses ki energy, agile strikes, bare-handed combat.",
        "subclasses": {
            "ascendant dragon": "breathes elemental energy, draconic wings, elemental strikes.",
            "astral self": "summons spectral arms, relies on wisdom, cosmic energy.",
            "drunken master": "unpredictable movement, disengages easily, redirects attacks.",
            "four elements": "bends elements, casts spells with ki, fiery strikes and watery whips.",
            "kensei": "weapon master, blends ki with swords and bows, elegant strikes.",
            "long death": "obsessed with mortality, feeds on death, incredibly hard to kill.",
            "mercy": "plague doctor, heals with a touch, inflicts toxic necrotic damage.",
            "open hand": "master of unarmed combat, trips and stuns opponents, self-healing.",
            "shadow": "ninja, teleports through darkness, stealth and infiltration.",
            "sun soul": "shoots bolts of radiant light, burning ki, ranged martial artist."
        }
    },
    "paladin": {
        "base": "holy knight, divine smite, heavy armor, auras of protection, heals by touch.",
        "subclasses": {
            "ancients": "green knight, protects nature, wards against spell damage.",
            "conquest": "terrifying presence, freezes enemies with fear, rules with an iron fist.",
            "crown": "sworn to a sovereign, redirects damage to self, challenges foes to duels.",
            "devotion": "classic holy knight, glowing weapons, turns the unholy.",
            "glory": "seeks heroic deeds, athletic prowess, inspires allies with glorious acts.",
            "oathbreaker": "fallen knight, commands undead, auras of hatred and darkness.",
            "redemption": "pacifist warrior, absorbs damage for others, highly persuasive.",
            "vengeance": "relentless avenger, hunts down foes, aggressive damage dealer.",
            "watchers": "guards against extraplanar threats, banishes fiends and aberrations."
        }
    },
    "ranger": {
        "base": "wilderness survivor, tracker, uses bows and two-weapon fighting, nature magic.",
        "subclasses": {
            "beast master": "fights alongside a loyal animal companion, bonded through nature.",
            "drakewarden": "bonded to a draconic spirit, breathes elements, rides a drake.",
            "fey wanderer": "fey magic, psychic damage, charming and frightening, highly charismatic.",
            "gloom stalker": "creature of darkness, deadly first strikes, invisible in shadows.",
            "horizon walker": "guards planar portals, teleports short distances, force damage strikes.",
            "hunter": "slayer of monsters, versatile combatant against hordes or giants.",
            "monster slayer": "hunts magical beasts, counterspells, exploits enemy weaknesses.",
            "swarmkeeper": "surrounded by nature spirits, moves enemies, bugs and fey creatures."
        }
    },
    "rogue": {
        "base": "stealthy skirmisher, sneak attacks, skilled with lockpicks, avoids detection.",
        "subclasses": {
            "arcane trickster": "magical thief, uses illusions and enchantments to steal and deceive.",
            "assassin": "deadly killer, poison expert, strikes from the shadows, surprise attacks.",
            "inquisitive": "master detective, reads intentions, insightful fighting, spots lies.",
            "mastermind": "tactical genius, mimics voices, directs allies in combat from afar.",
            "phantom": "communes with spirits, necrotic damage, steals knowledge from the dead.",
            "scout": "wilderness survivor, extremely mobile, skirmisher, expert tracker.",
            "soulknife": "manifests psychic blades, telepathic, perfects skills with psionics.",
            "swashbuckler": "dashing duelist, highly charismatic, maneuvers easily in melee.",
            "thief": "agile burglar, treasure hunter, climbs walls, disarms traps rapidly."
        }
    },
    "sorcerer": {
        "base": "innate spellcaster, metamagic, manipulates spell effects, charismatic.",
        "subclasses": {
            "aberrant mind": "psionic powers, telepathy, eldritch tentacles, cosmic horror.",
            "clockwork soul": "order magic, prevents advantages, summons constructs, cosmic gears.",
            "divine soul": "chosen by gods, combines cleric healing with sorcerer spells.",
            "draconic bloodline": "dragon scales, elemental affinity, breathes fire or lightning, wings.",
            "lunar sorcery": "moon magic, shifts between phases, radiant and shadow spells.",
            "shadow magic": "creatures of darkness, summons a hound of ill omen, resists death.",
            "storm sorcery": "controls wind and lightning, flies short distances, elemental aura.",
            "wild magic": "unpredictable chaos, surges of random magic effects, bends luck."
        }
    },
    "warlock": {
        "base": "pact magic, eldritch blast, granted power by otherworldly patrons, invocations.",
        "subclasses": {
            "archfey": "fey tricks, charming, teleporting, illusory defenses.",
            "celestial": "radiant energy, healing light, searing vengeance, angelic patron.",
            "fathomless": "oceanic magic, summons spectral tentacles, swims, breathes underwater.",
            "fiend": "hellish power, commands fire, gains temporary health on kills.",
            "genie": "elemental patron, rests inside a magical vessel, elemental strikes.",
            "great old one": "telepathic, mind-reading, psychic damage, eldritch horror patron.",
            "hexblade": "shadow magic warrior, curses foes, fights with conjured melee weapons.",
            "undead": "form of dread, terrifying presence, resists necrotic, spectral attacks.",
            "undying": "defies death, immune to disease, heals when successful at death saves."
        }
    },
    "wizard": {
        "base": "scholarly spellcaster, huge spellbook, rituals, master of arcane studies.",
        "subclasses": {
            "abjuration": "protective magic, arcane wards, counterspells, absorbs damage.",
            "bladesinging": "elven sword dancer, combines melee strikes with spellcasting, agile.",
            "chronurgy": "time magic, forces rerolls, freezes enemies in time.",
            "conjuration": "summons creatures, teleports, creates objects out of thin air.",
            "divination": "foresees the future, alters dice rolls, gathers distant intel.",
            "enchantment": "mind control, hypnotic gaze, alters memories, manipulates foes.",
            "evocation": "destructive magic, fireballs, sculpts spells around allies safely.",
            "graviturgy": "gravity magic, alters weight, black holes, forceful movement.",
            "illusion": "master of deception, creates lifelike mirages, alters reality temporarily.",
            "necromancy": "animates the dead, drains life force, commands zombie hordes.",
            "order of scribes": "awakened spellbook, changes spell damage types, magical scholar.",
            "transmutation": "changes matter, alchemical stones, alters physical forms.",
            "war magic": "battlefield tactician, combines abjuration defenses with evocation strikes."
        }
    },
    "artificer": {
        "base": "magical inventor, infuses items with magic, creates constructs and gadgets.",
        "subclasses": {
            "alchemist": "brews magical potions, heals allies, throws acid and fire.",
            "armorer": "wears magical power armor, acts as an impenetrable tank.",
            "artillerist": "magical cannons, firearms, explosive magic, wand slinger.",
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
    "underground cavern", "ruined temple", "floating island", "tavern basement",
    "underdark fungal forest", "coastal cliffs", "haunted mansion", "ruined castle",
    "bustling market", "gladiatorial arena", "underwater grotto", "giant's keep",
    "goblin war camp", "moving airship", "labyrinthine catacombs", "crystal caves",
    "desolate badlands", "abandoned monastery", "magical academy library",
    "clockwork laboratory", "sunken city ruins", "desecrated graveyard", "petrified forest"
]

PLANES = [
    "Astral Plane silver void", "Feywild enchanted forest", "Shadowfell domain of dread",
    "Nine Hells fiery battlefield", "City of Brass in the Elemental Plane of Fire",
    "Endless layers of the Abyss", "Mechanus clockwork gears", "Limbo chaotic storm",
    "Mount Celestia shining peaks", "Bytopia pastoral twin paradises",
    "Elysium peaceful meadows", "Beastlands wild forests", "Arborea passionate wilderness",
    "Ysgard heroic battlefields", "Pandemonium howling caverns", "Carceri red prison planet",
    "Hades gray wastes", "Gehenna volcanic steep slopes", "Acheron colliding iron cubes",
    "Arcadia orderly orchards", "Plane of Water endless ocean", "Plane of Air endless sky",
    "Plane of Earth deep subterranean", "Sigil city of doors", "Far Realm maddening geometry"
]

ATMOSPHERE = [
    "thick fog", "heavy rain", "pitch black", "cobwebs and dust",
    "smell of sulfur", "glowing arcane runes", "howling blizzard",
    "eerie silence", "bloodstains", "magical twilight", "overgrown with vines",
    "crumbling walls", "knee-deep mud", "oppressive heat", "toxic fumes",
    "echoing dripping water", "blinding sandstorm", "magical wild magic zone",
    "unnaturally cold", "scent of ozone", "floating debris from zero gravity",
    "shifting architecture", "deafening thunder", "spectral whispers", "ankle-deep ash",
    "bioluminescent flora", "choking smoke", "prismatic lighting", "magnetic anomalies",
    "complete absence of sound", "smell of rotting flesh", "swarming with insects"
]

ENEMY_FACTIONS = [
    "bandit highwaymen", "pirate mutineers", "doomsday cultists", "drow assassins",
    "undead horde", "vampire spawn", "necromancer and skeletons",
    "mind flayer colony", "beholder", "yuan-ti abominations", "giant spiders",
    "mimics and ropers", "young red dragon", "hag coven", "fire elementals",
    "abyssal demons", "goblin raiding party", "rogue artificer constructs",
    "orc warband", "hobgoblin legion", "mind flayer thralls", "githyanki raiding party",
    "githzerai monks", "gnoll pack", "kobold trap-makers", "sahuagin baron's guard",
    "kuo-toa zealots", "slaadi slavers", "rakshasa syndicate", "lycanthrope pack",
    "shadow fey court", "modron marching battalion", "elemental myrmidons",
    "cloaker and darkmantles", "myconid circle", "yuan-ti purebloods",
    "vampire lord's court", "lich's undead army", "duergar slavers", "troll scavengers"
]

ENEMY_ACTIONS = [
    "setting up an ambush", "guarding a locked chest", "conducting a dark ritual",
    "sleeping", "patrolling the area", "fighting a rival group",
    "interrogating a prisoner", "feasting on a corpse", "searching for intruders",
    "repairing their weapons", "hiding in the shadows", "worshipping an idol",
    "deciphering an ancient text", "harvesting poisonous fungi", "counting looted gold",
    "arguing over leadership", "torturing a captive", "constructing a siege weapon",
    "drawing summoning circles", "recovering from a previous battle",
    "playing crude gambling games", "fortifying their position", "tracking the players",
    "preparing a feast", "arguing with their familiar", "burying stolen treasure"
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