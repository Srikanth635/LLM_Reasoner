from pydantic import BaseModel
from typing import Dict, Any, Optional
from enum import Enum
import json

class Size(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    TINY = "tiny"
    HUGE = "huge"

class Length(str, Enum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"

class Width(str, Enum):
    NARROW = "narrow"
    MEDIUM = "medium"
    WIDE = "wide"

class Height(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    TALL = "tall"

class Volume(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class Shape(str, Enum):
    ROUND = "round"
    OVAL = "oval"
    SQUARE = "square"
    RECTANGULAR = "rectangular"
    CYLINDRICAL = "cylindrical"
    CONICAL = "conical"
    IRREGULAR = "irregular"
    FLAT = "flat"
    SPHERICAL = "spherical"
    CUBICAL = "cubical"
    HEMISPHERICAL = "hemispherical"

class Symmetry(str, Enum):
    RADIAL = "radial"
    BILATERAL = "bilateral"
    ASYMMETRIC = "asymmetric"
    NONE = "none"

class Color(str, Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"
    YELLOW = "yellow"
    ORANGE = "orange"
    PURPLE = "purple"
    BROWN = "brown"
    BLACK = "black"
    WHITE = "white"
    GREY = "grey"
    CLEAR = "clear"

class Texture(str, Enum):
    SMOOTH = "smooth"
    ROUGH = "rough"
    BUMPY = "bumpy"
    FUZZY = "fuzzy"
    PRICKLY = "prickly"
    SLIPPERY = "slippery"
    STICKY = "sticky"
    GRAINY = "grainy"
    LAYERED = "layered"
    FLAKY = "flaky"
    CRISP = "crisp"
    SOFT = "soft"
    HARD = "hard"
    WAXY = "waxy"
    POWDERY = "powdery"

class Pattern(str, Enum):
    SOLID = "solid"
    STRIPED = "striped"
    SPOTTED = "spotted"
    MARBLED = "marbled"
    CHECKED = "checked"
    FLORAL = "floral"
    GRAPHIC = "graphic"
    NONE = "none"

class Reflectance(str, Enum):
    GLOSSY = "glossy"
    MATTE = "matte"
    SHINY = "shiny"
    DULL = "dull"

class Transparency(str, Enum):
    OPAQUE = "opaque"
    TRANSLUCENT = "translucent"
    TRANSPARENT = "transparent"

class Material(str, Enum):
    METAL = "metal"
    PLASTIC = "plastic"
    CERAMIC = "ceramic"
    GLASS = "glass"
    WOOD = "wood"
    FABRIC = "fabric"
    PAPER = "paper"
    RUBBER = "rubber"
    SILICONE = "silicone"
    STONE = "stone"
    ORGANIC = "organic"

class Weight(str, Enum):
    LIGHT = "light"
    MEDIUM = "medium"
    HEAVY = "heavy"

class Density(str, Enum):
    LOW_DENSITY = "lowdensity"
    MEDIUM_DENSITY = "mediumdensity"
    HIGH_DENSITY = "highdensity"
    DENSE = "dense"

class Firmness(str, Enum):
    SOFT = "soft"
    MEDIUM = "medium"
    FIRM = "firm"
    HARD = "hard"
    RIGID = "rigid"
    SQUISHY = "squishy"
    BRITTLE = "brittle"

class Grip(str, Enum):
    SMOOTH = "smooth"
    TEXTURED = "textured"
    EASY_GRIP = "easygrip"
    SLIPPERY = "slippery"
    SECURE = "secure"

class Balance(str, Enum):
    BALANCED = "balanced"
    BLADE_HEAVY = "bladeheavy"
    HANDLE_HEAVY = "handleheavy"

class Handle(str, Enum):
    PRESENT = "present"
    NONE = "none"
    SINGLE = "single"
    DOUBLE = "double"
    LOOP = "loop"
    STRAIGHT = "straight"

class Blade(str, Enum):
    PRESENT = "present"
    NONE = "none"
    STRAIGHT = "straight"
    SERRATED = "serrated"
    CURVED = "curved"
    SHORT = "short"
    LONG = "long"
    SHARP = "sharp"
    DULL = "dull"

class Edge(str, Enum):
    PRESENT = "present"
    NONE = "none"
    SHARP = "sharp"
    DULL = "dull"
    STRAIGHT = "straight"
    CURVED = "curved"
    BEVELED = "beveled"
    ROUNDED = "rounded"

class Point(str, Enum):
    PRESENT = "present"
    NONE = "none"
    SHARP = "sharp"
    ROUNDED = "rounded"
    BLUNT = "blunt"

class Corners(str, Enum):
    PRESENT = "present"
    NONE = "none"
    SHARP = "sharp"
    ROUNDED = "rounded"

class Skin(str, Enum):
    PRESENT = "present"
    NONE = "none"
    THIN = "thin"
    THICK = "thick"
    SMOOTH = "smooth"
    ROUGH = "rough"
    PEELED = "peeled"

class Cleanliness(str, Enum):
    CLEAN = "clean"
    DIRTY = "dirty"
    WASHED = "washed"
    STICKY = "sticky"
    GREASY = "greasy"

class Condition(str, Enum):
    WHOLE = "whole"
    CUT = "cut"
    SLICED = "sliced"
    DICED = "diced"
    CHOPPED = "chopped"
    PEELED = "peeled"
    BRUISED = "bruised"
    BROKEN = "broken"
    CRACKED = "cracked"
    BENT = "bent"
    DAMAGED = "damaged"
    GOOD = "good"
    BAD = "bad"

class Intactness(str, Enum):
    INTACT = "intact"
    BROKEN = "broken"
    DAMAGED = "damaged"
    PARTIAL = "partial"

class Freshness(str, Enum):
    FRESH = "fresh"
    STALE = "stale"
    WILTING = "wilting"
    EXPIRED = "expired"

class Ripeness(str, Enum):
    UNRIPE = "unripe"
    RIPE = "ripe"
    OVERRIPE = "overripe"

class Dirt(str, Enum):
    NONE = "none"
    SOME = "some"
    HEAVY = "heavy"

class Count(str, Enum):
    SINGLE = "single"
    MULTIPLE = "multiple"
    FEW = "few"
    MANY = "many"

class Orientation(str, Enum):
    UPRIGHT = "upright"
    SIDEWAYS = "sideways"
    INVERTED = "inverted"
    ANGLED = "angled"

class Position(str, Enum):
    ON = "on"
    IN = "in"
    UNDER = "under"
    NEAR = "near"
    FAR = "far"
    LEFT = "left"
    RIGHT = "right"
    FRONT = "front"
    BACK = "back"
    CENTER = "center"
    EDGE = "edge"

class Odor(str, Enum):
    NONE = "none"
    MILD = "mild"
    STRONG = "strong"
    SWEET = "sweet"
    SOUR = "sour"
    PUNGENT = "pungent"
    AROMATIC = "aromatic"
    BURNT = "burnt"
    SPICY = "spicy"

class ActionTypes(str, Enum):
    PICKINGUP = "PickingUp"
    PLACING = "Placing"
    POURING = "Pouring"
    OPENING = "Opening"
    CLOSING = "Closing"
    CUTTING = "Cutting"