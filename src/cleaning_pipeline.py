"""
DataDose — Pharmaceutical Data Cleaning & Normalization Pipeline
================================================================
Rule-based cleaning engine for drug ingredient data.
Prepares medication records for downstream drug interaction detection.

Usage:
    python src/cleaning_pipeline.py

Configuration:
    Edit BASE_DIR, INPUT_FILE, OUTPUT_FILE, LOG_FILE below to match your paths.
"""

import pandas as pd
import numpy as np
import re
import os
from datetime import datetime

# =============================================================================
# Configuration
# =============================================================================
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE  = os.path.join(BASE_DIR, 'data', 'raw', 'DataDoseDataset.csv')
OUTPUT_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'DataDoseDataset_CleanedDrugs.csv')
LOG_FILE    = os.path.join(BASE_DIR, 'logs', 'cleaning_logFinal.txt')

# =============================================================================
# P0 — Immutable Principles
# P0.1: Synonym merging is FORBIDDEN (paracetamol ↔ acetaminophen must stay as-is)
# P0.2: Duplicate rows across the dataset are allowed; only intra-row deduplication
# =============================================================================

# =============================================================================
# Encoded Token Decoder
# Handles __INGxxxx__ placeholders left by upstream encoding corruption.
# =============================================================================
ENCODED_TOKEN_MAP = {
    "__ING0024__": "vita",   # __ING0024__mins       → vitamins (filtered later)
    "__ING0035__": "",       # __ING0035__2           → bare "2" (filtered later)
    "__ING0055__": "iron",   # sp__ING0055__olactone → spironolactone
}

# =============================================================================
# Garbage Terms (Exact-match only — no substring matching)
# =============================================================================
GARBAGE_LIST = [
    "invalid", "test", "unknown", "no active ingredient", "pending", "deleted",
    "n/a", "not available", "natural source", "mixed",
    "coming soon", "special", "bitten", "mental",
    "herbal formula", "regurgitation milk formula", "gasmin odor",
    "ethyhexyl", "stearly alcohol", "silicones",
    "glycerol stearate", "octadeceny ammonium",
    "distearoylethyl hydroxyethylmonium methosulfate",
    "capramidopropylbetaine", "capryl", "cocoamidopropyl betaine",
    "water", "aqua",
    "dr ey t", "dr ey",
    "gereinigter honig", "selected theraputically active",
    "theraputically active", "gereinigter", "honig",
    "selected theraputically",
    "vitamins", "vita", "350m",
]
GARBAGE_EXACT = set(x.strip().lower() for x in GARBAGE_LIST)

# Short tokens that are garbage ONLY when they appear as a complete standalone token
GARBAGE_TOKENS_EXACT = {
    "na", "n/a", "amin", "amins", "type", "formula",
    "other", "high", "pre", "mixed", "special",
    "as", "ivay", "potat", "len",
    "2",
    "vitamin",
}

# =============================================================================
# NON-DRUG TOKENS — Supplements/marketing terms that are NOT active ingredients
# =============================================================================
NON_DRUG_TOKENS = {
    "q10", "coq10", "q 10",
    "royal jelly", "propolis", "bee pollen", "bee wax",
    "antioxidants", "antioxidant",
    "green tea extract", "grape seed extract", "pine bark extract",
    "ginkgo biloba", "ginseng",
    "honey", "beeswax", "aloe vera", "aloe",
    "herbal extract", "plant extract", "natural extract",
    "amino acids blend", "protein blend", "mineral blend",
    "turmeric", "curcumin", "ginger", "ginger extract",
    "garlic", "garlic extract", "garlic powder",
    "cinnamon", "cinnamon extract",
    "black seed", "black seed oil", "nigella sativa",
    "evening primrose", "evening primrose oil",
    "flaxseed", "flaxseed oil", "fish oil",
    "peppermint", "peppermint oil",
    "chamomile", "chamomile extract",
    "echinacea", "valerian", "ashwagandha",
    "msm", "methylsulfonylmethane",
    "lutein", "zeaxanthin", "lycopene", "astaxanthin",
    "resveratrol", "quercetin",
    "spirulina", "chlorella",
    "milk thistle",
}

# =============================================================================
# R4 — Spell-fix Dictionary (ONLY true typos — NO synonym merging per P0.1!)
# =============================================================================
SPELL_FIX = {
    "benzylpenicillin sodiium":  "benzylpenicillin sodium",
    "cholorohexidine":           "chlorhexidine",
    "chlorohexidine":            "chlorhexidine",
    "chlorohexidin":             "chlorhexidine",
    "nitrofurantion":            "nitrofurantoin",
    "immunoglobulins":           "immunoglobulin",
    "panthenoll":                "panthenol",
    "pantheno":                  "panthenol",
    "pilocarpin":                "pilocarpine",
    "macrophages":               "macrophage",
    "macrofage":                 "macrophage",
    "olanzapin":                 "olanzapine",
    "diclofienac":               "diclofenac",
    "sildeanfil":                "sildenafil",
    "sindalfil":                 "sildenafil",
    "digoxine":                  "digoxin",
}

# =============================================================================
# Manual Replacements — Applied BEFORE generic split
# =============================================================================
PLAIN_REPLACEMENTS = {
    "vit.":          "vitamin ",
    "vitamin b complex": (
        "thiamine + riboflavin + niacin + pantothenic acid + "
        "pyridoxine + biotin + folic acid + cobalamin"
    ),
    "b complex": (
        "thiamine + riboflavin + niacin + pantothenic acid + "
        "pyridoxine + biotin + folic acid + cobalamin"
    ),
}

# Regex word-boundary replacements for B-vitamin codes (order: longest first)
BVITAMIN_REGEX = [
    (re.compile(r'\bvit\b\.?'),  "vitamin "),
    (re.compile(r'\bb12\b'),     "cobalamin"),
    (re.compile(r'\bb9\b'),      "folic acid"),
    (re.compile(r'\bb7\b'),      "biotin"),
    (re.compile(r'\bb6\b'),      "pyridoxine"),
    (re.compile(r'\bb5\b'),      "pantothenic acid"),
    (re.compile(r'\bb3\b'),      "niacin"),
    (re.compile(r'\bb2\b'),      "riboflavin"),
    (re.compile(r'\bb1\b'),      "thiamine"),
]

REPLACEMENTS = PLAIN_REPLACEMENTS

# =============================================================================
# Known Ingredient Vocabulary — Used for garbage phrase & unknown token detection
# =============================================================================
KNOWN_INGREDIENT_KEYWORDS = {
    "vitamin", "acid", "calcium", "magnesium", "zinc", "iron", "sodium",
    "potassium", "chloride", "oxide", "hydrochloride", "sulfate", "phosphate",
    "gluconate", "citrate", "acetate", "lactate", "carbonate", "nitrate",
    "immunoglobulin", "albumin", "insulin", "heparin", "factor", "hormone",
    "enzyme", "extract", "compound", "complex", "analog", "analogue",
    "colony", "stimulating", "granulocyte", "macrophage", "ketoanalogue",
    "histidine", "lysine", "threonine", "tryptophan", "tyrosine", "amino",
    "iodo", "chloro", "hydroxy", "quinoline", "biotin", "niacin", "riboflavin",
    "pantothenic", "pyridoxine", "thiamine", "cobalamin", "folic", "selenium",
    "manganese", "copper", "boron", "chromium", "molybdenum", "fluoride",
    "iodochlorohydroxyquinoline", "panthenol", "pilocarpine", "omega",
    "retinol", "tocopherol", "ascorbic", "cholecalciferol", "ergocalciferol",
    "menadione", "phytomenadione", "alpha", "beta", "gamma", "delta",
    "methionine", "cysteine", "arginine", "leucine", "isoleucine", "valine",
    "alanine", "glycine", "proline", "serine", "glutamine", "asparagine",
    "aspartate", "glutamate", "phenylalanine",
    "coenzyme", "ubiquinone", "carnitine", "taurine", "inositol", "choline",
    "lipoic", "rutin", "hesperidin", "quercetin", "flavonoid",
    "glucosamine", "chondroitin", "collagen", "hyaluronic",
    "probiotic", "prebiotic", "lactobacillus", "bifidobacterium",
    "interferon", "erythropoietin", "filgrastim", "pegfilgrastim",
    "antitoxin", "antivenom", "vaccine",
    "paracetamol", "acetaminophen", "ibuprofen", "aspirin", "caffeine",
    "codeine", "morphine", "tramadol", "diclofenac", "naproxen",
    "amoxicillin", "ampicillin", "penicillin", "cephalexin", "azithromycin",
    "ciprofloxacin", "metronidazole", "doxycycline", "tetracycline",
    "metformin", "glibenclamide", "atorvastatin", "simvastatin",
    "amlodipine", "enalapril", "losartan", "hydrochlorothiazide", "furosemide",
    "omeprazole", "ranitidine", "metoclopramide", "domperidone", "ondansetron",
    "salbutamol", "terbutaline", "beclomethasone", "fluticasone", "ipratropium",
    "prednisolone", "dexamethasone", "hydrocortisone", "betamethasone",
    "loratadine", "cetirizine", "diphenhydramine", "promethazine",
    "diazepam", "alprazolam", "lorazepam", "clonazepam", "phenobarbital",
    "haloperidol", "risperidone", "olanzapine", "quetiapine", "aripiprazole",
    "fluoxetine", "sertraline", "paroxetine", "escitalopram", "venlafaxine",
    "levothyroxine", "propylthiouracil", "methimazole",
    "warfarin", "enoxaparin", "clopidogrel",
    "cyclosporine", "tacrolimus", "mycophenolate", "azathioprine",
    "methotrexate", "cyclophosphamide", "doxorubicin", "fluorouracil",
    "sildenafil", "tadalafil", "testosterone", "estradiol", "progesterone",
    "spironolactone", "dandelion", "silymarin", "iodine",
    "poliomyelitis", "inactivated", "attenuated", "poliovirus",
    "willebrand", "von",
}

# =============================================================================
# R5 — Cosmetic / Personal-Care Terms
# =============================================================================
COSMETIC_TERMS = {
    "cream", "shampoo", "lotion", "styling", "smooth", "hair", "gel",
    "serum", "moisturizer", "conditioner", "spray", "foam", "mask",
    "scrub", "toner", "cleanser", "balm", "wax", "polish",
    "blush", "foundation", "lipstick", "mascara", "perfume",
    "fragrance", "deodorant", "sunscreen", "exfoliant", "primer",
    "scalp", "skin", "whitening", "regen", "matrix", "photostable",
    "uva", "uvb",
}

# =============================================================================
# R6 — Vague Category Terms
# =============================================================================
VAGUE_CATEGORY_EXACT = {
    "minerals", "elements", "omega", "ors", "carbohydrates",
    "proteins", "multivitamin", "multivitamins",
    "vitamins and minerals", "vitamins", "trace elements",
}

# =============================================================================
# R7 — Truncated Token Detection
# =============================================================================
TRUNCATED_TOKENS = {
    "ethinyl", "mono", "hydro", "peg", "poly",
    "micronized alpha", "micronized", "dehydro", "desoxy", "nor",
}

# =============================================================================
# R8 — Unknown Token Detection
# =============================================================================
SHORT_VALID_TOKENS = {
    "a", "c", "d", "e", "k",
    "d2", "d3", "k1", "k2", "k3",
    "b1", "b2", "b3", "b5", "b6", "b7", "b9", "b12",
    "ors", "rna", "dna", "hiv", "ige", "igg", "iga", "igm",
    "atp", "adp", "nad", "gmp", "amp",
    "viii", "vii", "vi", "iv", "xii", "xiii",
}

UNKNOWN_TOKEN_PATTERN = re.compile(
    r'^[a-z]{1,3}\d+$'
    r'|^[a-z0-9]{1,3}\s[a-z0-9]{1,2}$'
)

VALID_VITAMIN_LETTERS = {
    "a", "c", "d", "e", "k",
    "d2", "d3", "k1", "k2", "k3",
    "b1", "b2", "b3", "b5", "b6", "b7", "b9", "b12",
}

# =============================================================================
# Pattern for inserting '+' between unseparated known ingredients
# =============================================================================
_UNSEP = (
    r'calcium|magnesium|zinc|iron|selenium|manganese|copper|boron|chromium|'
    r'molybdenum|fluoride|biotin|niacin|riboflavin|thiamine|pyridoxine|'
    r'pantothenic|folic|cobalamin|lysine|histidine|threonine|tryptophan|'
    r'tyrosine|iodine|iodo|granulocyte|albumin|insulin|heparin|collagen|'
    r'glucosamine|chondroitin|carnitine|taurine|inositol|choline|'
    r'ubiquinone|rutin|hesperidin|quercetin|lipoic|coenzyme|'
    r'lactobacillus|bifidobacterium|probiotic|prebiotic|'
    r'paracetamol|acetaminophen|ibuprofen|aspirin|caffeine|codeine|'
    r'amoxicillin|ampicillin|penicillin|ciprofloxacin|metronidazole|'
    r'metformin|atorvastatin|simvastatin|amlodipine|enalapril|losartan|'
    r'omeprazole|ranitidine|ondansetron|salbutamol|terbutaline|'
    r'prednisolone|dexamethasone|hydrocortisone|betamethasone|loratadine|'
    r'cetirizine|diazepam|alprazolam|fluoxetine|sertraline|levothyroxine|'
    r'warfarin|cyclosporine|methotrexate|sildenafil|testosterone|estradiol|'
    r'progesterone|spironolactone|dandelion|silymarin|retinol|tocopherol|'
    r'ascorbic|cholecalciferol|menadione'
)
UNSEPARATED_SPLIT_PATTERN = re.compile(
    r'(?<=[a-z\d])\s+(?=(' + _UNSEP + r')\b)'
)

# R3 — Dosage unit pattern
DOSE_UNIT_PATTERN = re.compile(
    r'\d+(\.\d+)?\s*(mg|g|gm|mcg|µg|ug|iu|i\s*u|miu|ml|%|units?|tabs?|caps?|amp|vial)\b',
    re.IGNORECASE
)

# R3.1 — Leading numeric token pattern
LEADING_NUMBER_PATTERN = re.compile(
    r'^\s*[\d\s\.]+\s*'
    r'(mg|g|gm|mcg|µg|ug|iu|i\s*u|miu|ml|%|units?|tabs?|caps?|amp|vial)?\s*$',
    re.IGNORECASE
)


# =============================================================================
# Logging
# =============================================================================
def log_message(message):
    """Log messages to console and file with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"[{timestamp}] {message}"
    print(msg)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass


# =============================================================================
# Step 0 — Encoded Token Decoder
# =============================================================================
def decode_encoded_tokens(text):
    """Replace __INGxxxx__ placeholders with known decoded values."""
    if not isinstance(text, str):
        return text
    for token, replacement in ENCODED_TOKEN_MAP.items():
        text = text.replace(token, replacement)
    text = re.sub(r'__[A-Z]+\d+__', ' ', text)
    return text


# =============================================================================
# R4 — Spell Correction
# =============================================================================
def apply_spell_fix(text):
    """Apply spell corrections using word-boundary regex. No synonym merging (P0.1)."""
    if not isinstance(text, str):
        return text
    for wrong, right in sorted(SPELL_FIX.items(), key=lambda x: -len(x[0])):
        if wrong in text:
            text = re.sub(r'\b' + re.escape(wrong) + r'\b', right, text)
    return text


# =============================================================================
# Garbage Detection
# =============================================================================
def is_garbage_token(token):
    """Return True if a single ingredient token should be discarded."""
    t = token.strip().lower()
    if not t or len(t) <= 2:
        return True
    if t in GARBAGE_EXACT:
        return True
    if t in GARBAGE_TOKENS_EXACT:
        return True
    if t in NON_DRUG_TOKENS:
        return True
    for g in GARBAGE_EXACT:
        if len(g) >= 8 and g in t:
            return True
    return False


def is_likely_garbage_phrase(text):
    """Detect multi-word free-text with NO recognizable ingredient vocabulary."""
    words = text.lower().split()
    if len(words) < 3:
        return False
    matches = sum(
        1 for w in words
        if any(kw in w for kw in KNOWN_INGREDIENT_KEYWORDS)
    )
    return matches == 0


# =============================================================================
# R5 — Cosmetic Entry Detection
# =============================================================================
def is_cosmetic_entry(text):
    """Return True if the entry looks like a cosmetic / personal-care product."""
    if not text:
        return False
    words = set(text.lower().split())
    return len(words & COSMETIC_TERMS) >= 2


# =============================================================================
# R6 — Vague Category Detection
# =============================================================================
def is_vague_category(text):
    """Return True if the entire entry is a vague/non-specific category."""
    if not text:
        return False
    t = text.strip().lower()
    if t in VAGUE_CATEGORY_EXACT:
        return True
    tokens = [p.strip() for p in t.split('+')]
    return all(tok in VAGUE_CATEGORY_EXACT for tok in tokens if tok)


# =============================================================================
# R7 — Truncated Token Detection
# =============================================================================
def has_truncated_token(parts):
    """Return True if any part looks like a truncated/incomplete ingredient."""
    for p in parts:
        p_lower = p.strip().lower()
        if p_lower in TRUNCATED_TOKENS:
            return True
        if re.match(r'^(hydro|mono|poly|peg|dehydro|desoxy|nor)$', p_lower):
            return True
    return False


# =============================================================================
# R8 — Unknown Token Classification
# =============================================================================
def classify_token(token):
    """Returns 'valid' or 'unknown' for a given token."""
    t = token.strip().lower()
    if not t:
        return 'valid'
    if t in SHORT_VALID_TOKENS:
        return 'valid'
    if t.startswith('vitamin '):
        return 'valid'
    for kw in KNOWN_INGREDIENT_KEYWORDS:
        if kw in t:
            return 'valid'
    if t in SPELL_FIX.values():
        return 'valid'
    if UNKNOWN_TOKEN_PATTERN.match(t):
        return 'unknown'
    if len(t) <= 3 and t not in SHORT_VALID_TOKENS:
        return 'unknown'
    return 'valid'


# =============================================================================
# R11 — Omega Format Normalization
# =============================================================================
def normalize_omega(text):
    """R11: Split omega combos into separate + tokens."""
    def expand_omega_multi(m):
        nums = m.group(1).split('-')
        return ' + '.join('omega ' + n for n in nums)

    text = re.sub(r'\bomega-(\d+(?:-\d+)+)\b', expand_omega_multi, text)
    text = re.sub(r'\bomega-(\d+)\b', r'omega \1', text)

    def expand_omega_spaced(m):
        nums = m.group(1).strip().split()
        if len(nums) > 1:
            return ' + '.join('omega ' + n for n in nums)
        return 'omega ' + nums[0]

    text = re.sub(r'\bomega\s+(\d+(?:\s+\d+)+)\b', expand_omega_spaced, text)
    return text


# =============================================================================
# Text Normalization
# =============================================================================
def normalize_text(text):
    """Full normalization pipeline for a single text entry."""
    if pd.isna(text) or not isinstance(text, str):
        return None

    text = text.strip()
    if not text or text.lower() in GARBAGE_EXACT:
        return None

    text = text.lower()

    # R4 — Spell fix
    text = apply_spell_fix(text)

    # Plain replacements
    for wrong, right in PLAIN_REPLACEMENTS.items():
        if wrong in text:
            text = text.replace(wrong, right)

    # B-vitamin codes
    for pattern, right in BVITAMIN_REGEX:
        text = pattern.sub(right, text)

    # R11 — Omega normalization
    text = normalize_omega(text)

    # Vitamin abbreviations
    text = re.sub(r'\bvit\.?\s+', 'vitamin ', text)

    # Remove 'amin/amins' artifacts
    text = re.sub(r'\bvitamin\s+amins?\b', 'vitamin', text)
    text = re.sub(r'\bamins?\b', '', text)

    # Brackets → separators
    text = re.sub(r'\(([^)]*)\)', r' + \1', text)

    # Insert '+' before 'vitamin' when missing
    text = re.sub(r'(?<!\+)\s+(vitamin\s)', r' + \1', text)

    # Insert '+' between known unseparated ingredients
    text = UNSEPARATED_SPLIT_PATTERN.sub(' + ', text)

    # R1 — Normalize all separators → ' + '
    text = re.sub(r'(--|–|-|/|,|;|\\|&| and | with |&|\+)', ' + ', text)

    # R3 — Remove dosage strengths
    _TYPE_MEDICAL_CONTEXT = {
        "poliomyelitis", "poliovirus", "vaccine", "hepatitis",
        "diphtheria", "pertussis", "meningitis", "rotavirus",
        "herpes", "adenovirus", "coronavirus", "influenza",
        "dengue", "rabies", "typhoid", "cholera",
        "collagen", "diabetes",
    }
    _full_has_type_context = any(kw in text for kw in _TYPE_MEDICAL_CONTEXT)
    _type_protected = {}

    def _protect_type_smart(m):
        if _full_has_type_context:
            key = "__TYPEPROT" + str(len(_type_protected)) + "__"
            _type_protected[key] = m.group(0).strip()
            return key
        return " "

    text = re.sub(r'\btype\s+\d+\b', _protect_type_smart, text)
    text = DOSE_UNIT_PATTERN.sub(' ', text)
    text = re.sub(r'\b\d{2,}\b', ' ', text)
    text = re.sub(r'(?<=[a-z])\s+\d+(?:\s+\d+)*\s*$', ' ', text)
    text = re.sub(r'(?<=[a-z])\s+0\s+\d+', ' ', text)
    text = re.sub(r'(?<=[a-z])\s+\d\s+\d+\b', ' ', text)

    # Protect omega N and type N
    text = re.sub(r'\bomega\s+(\d)\b', r'omega__OMGPROT__\1', text)
    text = re.sub(r'\btype\s+(\d)\b', r'type__TYPROT__\1', text)
    text = re.sub(r'(?<=[a-z])\s+\d\b', ' ', text)
    text = text.replace('omega__OMGPROT__', 'omega ')
    text = text.replace('type__TYPROT__', 'type ')

    for key, val in _type_protected.items():
        text = text.replace(key, val)

    # Remove special characters
    text = re.sub(r'[^a-z0-9+\s]', ' ', text)

    # Clean whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\s*\+\s*', ' + ', text).strip()
    text = text.strip('+ ')

    return text if text else None


# =============================================================================
# Vitamin Shortcut Expansion
# =============================================================================
def expand_vitamin_shortcuts(parts, original_text):
    """Expand lone vitamin-letter tokens: c → vitamin c, d3 → vitamin d3, etc."""
    out = []
    has_vitamin = "vitamin" in (original_text or "")

    for p in parts:
        p = p.strip()
        if not p:
            continue
        if p.startswith("vitamin "):
            out.append(p)
            continue
        if has_vitamin and p in VALID_VITAMIN_LETTERS:
            out.append(f"vitamin {p}")
        else:
            out.append(p)

    return out


# =============================================================================
# R3.1 — Remove Leading Numeric Tokens
# =============================================================================
def remove_leading_numeric_tokens(parts):
    """R3.1: Drop tokens from the START that are purely numeric or dose-only."""
    while parts:
        first = parts[0].strip()
        if LEADING_NUMBER_PATTERN.match(first):
            parts = parts[1:]
        elif re.match(r'^\d+$', first):
            parts = parts[1:]
        else:
            break
    return parts


# =============================================================================
# Clean Individual Ingredients List
# =============================================================================
def clean_ingredient_list(parts, original_text):
    """Clean ingredient tokens: expand, filter, classify, deduplicate, sort."""
    parts = expand_vitamin_shortcuts(parts, original_text)
    parts = remove_leading_numeric_tokens(parts)

    cleaned     = []
    token_flags = []

    for ingredient in parts:
        ingredient = re.sub(r'\s+', ' ', ingredient).strip()
        ingredient = re.sub(r'\bamin[s]?\b', '', ingredient).strip()
        ingredient = re.sub(r'\b([a-z]{4,})\d+$', r'\1', ingredient)
        ingredient = re.sub(r'\s+', ' ', ingredient).strip()

        if is_garbage_token(ingredient):
            continue
        if len(ingredient) <= 2:
            continue

        token_class = classify_token(ingredient)
        if token_class == 'unknown':
            token_flags.append((ingredient, 'UNKNOWN'))

        cleaned.append(ingredient)

    return sorted(set(cleaned)), token_flags


# =============================================================================
# Main Per-Row Cleaning Function
# =============================================================================
def clean_active_ingredient(text):
    """Complete per-row pipeline: decode → normalize → filter → clean → validate."""
    row_flags      = []
    unknown_tokens = []

    text = decode_encoded_tokens(text)
    normalized = normalize_text(text)
    if normalized is None:
        return {"result": None, "row_flag": "", "unknown_tokens": []}

    if is_likely_garbage_phrase(normalized):
        return {"result": None, "row_flag": "", "unknown_tokens": []}

    if is_cosmetic_entry(normalized):
        return {"result": None, "row_flag": "COSMETIC", "unknown_tokens": []}

    parts = [p.strip() for p in normalized.split('+')]
    cleaned_parts, token_flags = clean_ingredient_list(parts, normalized)

    if not cleaned_parts:
        return {"result": None, "row_flag": "", "unknown_tokens": []}

    joined = " + ".join(cleaned_parts)
    if is_vague_category(joined):
        row_flags.append("VAGUE_CATEGORY")

    if has_truncated_token(cleaned_parts):
        row_flags.append("TRUNCATED")

    if token_flags:
        unknown_tokens = [t for t, _ in token_flags]
        row_flags.append("HAS_UNKNOWN_TOKENS")

    if cleaned_parts and re.match(r'^\d+', cleaned_parts[0]):
        row_flags.append("LEADING_NUMBER_UNRESOLVED")

    return {
        "result":         joined,
        "row_flag":       ",".join(row_flags) if row_flags else "",
        "unknown_tokens": unknown_tokens,
    }


# =============================================================================
# Main Pipeline
# =============================================================================
def clean_drug_ingredients(input_path, output_path):
    """Full drug-ingredient cleaning pipeline with logging and statistics."""
    log_message("=" * 70)
    log_message("DRUG INGREDIENT CLEANING PIPELINE v2 - STARTED")
    log_message("=" * 70)

    log_message(f"Loading data from: {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    initial_count = len(df)
    log_message(f"Loaded {initial_count:,} rows")
    log_message(f"Available columns: {list(df.columns)}")

    # Find ingredient column
    df.columns = df.columns.str.strip()
    possible_cols = [
        'ActiveIngredient', 'activeingredient', 'active_ingredient',
        'Generic Name', 'generic name', 'GenericName',
        'Ingredients', 'ingredients',
    ]

    col_name = None
    for col in possible_cols:
        if col in df.columns:
            col_name = col
            break

    if col_name is None:
        col_lower_map = {c.lower(): c for c in df.columns}
        for col in possible_cols:
            if col.lower() in col_lower_map:
                col_name = col_lower_map[col.lower()]
                break

    if col_name is None:
        raise ValueError(
            f"Active ingredient column not found. "
            f"Available columns: {list(df.columns)}"
        )

    log_message(f"Found ingredient column: '{col_name}'")

    # Audit encoded tokens
    encoded_mask  = df[col_name].str.contains(r'__[A-Z]+\d+__', na=False, regex=True)
    encoded_count = encoded_mask.sum()
    if encoded_count > 0:
        log_message(f"Found {encoded_count:,} rows with encoded tokens — decoding...")
        top_tokens = (
            df[encoded_mask][col_name]
            .str.findall(r'__[A-Z]+\d+__')
            .explode()
            .value_counts()
            .head(20)
        )
        log_message("Top encoded tokens:\n" + top_tokens.to_string())
    else:
        log_message("No encoded tokens detected.")

    # Apply cleaning
    log_message("Starting cleaning process...")
    cleaning_results = df[col_name].apply(clean_active_ingredient)

    df['activeingredient_clean'] = cleaning_results.apply(lambda x: x['result'])
    df['row_flag']               = cleaning_results.apply(lambda x: x['row_flag'])
    df['unknown_tokens']         = cleaning_results.apply(
        lambda x: "|".join(x['unknown_tokens']) if x['unknown_tokens'] else ""
    )

    df['Graph_Node_Ingredient'] = df['activeingredient_clean']
    df['ingredient_count'] = df['Graph_Node_Ingredient'].apply(
        lambda x: len(x.split(' + ')) if pd.notna(x) else 0
    )
    df['is_combination'] = df['ingredient_count'] > 1
    df['combo_type'] = df['ingredient_count'].apply(
        lambda n: 'single' if n == 1 else ('combo' if n > 1 else np.nan)
    )

    # Filter invalid / flagged rows
    df_valid = df[
        df['Graph_Node_Ingredient'].notna() &
        (df['ingredient_count'] > 0) &
        (df['row_flag'] == "")
    ].reset_index(drop=True)

    removed_count = initial_count - len(df_valid)

    # Save
    cols_to_drop = [c for c in ['row_flag', 'unknown_tokens'] if c in df_valid.columns]
    df_final = df_valid.drop(columns=cols_to_drop)
    df_final.to_csv(output_path, index=False, encoding='utf-8')

    # Summary
    log_message("")
    log_message("=" * 70)
    log_message("CLEANING COMPLETED SUCCESSFULLY!")
    log_message("=" * 70)
    log_message(f"Total rows processed:     {initial_count:,}")
    log_message(f"Valid rows retained:      {len(df_valid):,}")
    log_message(f"Invalid rows removed:     {removed_count:,}  "
                f"({removed_count / initial_count * 100:.1f}%)")
    log_message(f"Single ingredients:       {(df_valid['combo_type'] == 'single').sum():,}")
    log_message(f"Combination drugs:        {(df_valid['combo_type'] == 'combo').sum():,}")
    log_message(f"Average ingredients:      {df_valid['ingredient_count'].mean():.2f}")
    log_message(f"Max ingredients in combo: {df_valid['ingredient_count'].max()}")

    flag_series = df_valid['row_flag'].str.split(',', expand=True).stack()
    flag_counts = flag_series[flag_series != ''].value_counts()
    if not flag_counts.empty:
        log_message("\nRow flag summary (all should be empty — shown for debug):")
        for flag, cnt in flag_counts.items():
            log_message(f"  {flag:<35s}: {cnt:,}")
    else:
        log_message("\nNo flagged rows in output (all clean).")

    log_message("")
    log_message(f"Output saved to: {output_path}")
    log_message("=" * 70)

    return df_valid


# =============================================================================
# Sanity-Check Helper
# =============================================================================
def test_samples():
    """Run the cleaning function on all known edge-case samples and print results."""
    samples = [
        ("selected theraputically active gereinigter honig", None),
        ("biotin + folic acid + iron vitamin c folic acid vitamin thiamine + niacin + pantothenic acid + pyridoxine + riboflavin",
         "biotin + folic acid + iron + niacin + pantothenic acid + pyridoxine + riboflavin + thiamine + vitamin c"),
        ("human normal immunoglobulins", "human normal immunoglobulin"),
        ("iodochlorohydroxyquinoline", "iodochlorohydroxyquinoline"),
        ("dr ey t", None),
        ("calcium vitamin d3 vitamin k2 zinc boron copper manganese selenium magnesium",
         "boron + calcium + copper + magnesium + manganese + selenium + vitamin d3 + vitamin k2 + zinc"),
        ("alpha ketoanalogue of amino acids + histidine + lysine + threonine + tryptophan + tyrosine",
         "alpha ketoanalogue of amino acids + histidine + lysine + threonine + tryptophan + tyrosine"),
        ("granulocyte macrofage colony stimulating factor",
         "granulocyte macrophage colony stimulating factor"),
        ("__ING0035__2 + dandelion + folic acid + selenium + silymarin + vitamin c + vitamin e + zinc",
         "dandelion + folic acid + selenium + silymarin + vitamin c + vitamin e + zinc"),
        ("__ING0024__mins + __ING0035__2 + copper + folic acid + iodine + iron + niacin + pyridoxine + riboflavin + selenium + thiamine + zinc",
         "copper + folic acid + iodine + iron + niacin + pyridoxine + riboflavin + selenium + thiamine + zinc"),
        ("350m + cream + hair + smooth + styling", None),
        ("sp__ING0055__olactone", "spironolactone"),
        ("__ING0024__mins", None),
        ("omega-3 + vitamin e", "omega 3 + vitamin e"),
        ("omega-3-6-9 + vitamin c", "omega 3 + omega 6 + omega 9 + vitamin c"),
        ("150 + alpha + folic acid + iron", "alpha + folic acid + iron"),
        ("1000 + folic acid + vitamin b12", "cobalamin + folic acid"),
        ("cholorohexidine", "chlorhexidine"),
        ("digoxine", "digoxin"),
        ("panthenoll", "panthenol"),
        ("paracetamol", "paracetamol"),
        ("acetaminophen", "acetaminophen"),
        ("cream + hair + smooth + styling", None),
    ]

    print("\n" + "=" * 80)
    print("SANITY CHECK — EDGE CASE SAMPLES")
    print("=" * 80)

    passed = failed = 0

    for raw, expected in samples:
        res    = clean_active_ingredient(raw)
        result = res['result']
        flag   = res['row_flag']

        ok   = (result is None and expected is None) or (result == expected)
        icon = "PASS" if ok else "FAIL"
        passed += ok
        failed += (not ok)

        print(f"\n[{icon}]")
        print(f"  INPUT   : {raw}")
        print(f"  OUTPUT  : {result}")
        print(f"  FLAGS   : {flag or '(none)'}")
        if not ok:
            print(f"  EXPECTED: {expected}")

    print("\n" + "=" * 80)
    print(f"Results: {passed} passed, {failed} failed out of {len(samples)} tests")
    print("=" * 80 + "\n")
    return failed == 0


# =============================================================================
# Encoded Token Audit Helper
# =============================================================================
def audit_encoded_tokens(input_path):
    """Scan the raw dataset and print all unique encoded tokens found."""
    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip()

    possible_cols = [
        'ActiveIngredient', 'activeingredient', 'active_ingredient',
        'Generic Name', 'generic name', 'GenericName',
        'Ingredients', 'ingredients',
    ]
    col_name = None
    for col in possible_cols:
        if col in df.columns:
            col_name = col
            break
    if col_name is None:
        col_lower_map = {c.lower(): c for c in df.columns}
        for col in possible_cols:
            if col.lower() in col_lower_map:
                col_name = col_lower_map[col.lower()]
                break
    if col_name is None:
        print("Could not find ingredient column.")
        return

    mask         = df[col_name].str.contains(r'__[A-Z]+\d+__', na=False, regex=True)
    affected     = df[mask][col_name]
    all_tokens   = affected.str.findall(r'__[A-Z]+\d+__').explode()
    token_counts = all_tokens.value_counts()

    print("\n" + "=" * 70)
    print("ENCODED TOKEN AUDIT")
    print("=" * 70)
    print(f"Rows with encoded tokens: {mask.sum():,}")
    print(f"Unique token types:       {len(token_counts)}\n")
    print(token_counts.to_string())
    print("\n--- EXAMPLE ROWS PER TOKEN ---")
    for token in token_counts.index:
        examples = df[
            df[col_name].str.contains(re.escape(token), na=False)
        ][col_name].head(3).tolist()
        print(f"\n{token}  (count={token_counts[token]})")
        for ex in examples:
            print(f"  -> {ex}")
    print("=" * 70 + "\n")


# =============================================================================
# Entry Point
# =============================================================================
if __name__ == "__main__":
    try:
        # Clear previous log
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)

        # Sanity check
        all_passed = test_samples()
        if not all_passed:
            print("WARNING: Some sanity checks failed. Review before proceeding.\n")

        # Run full pipeline
        df_result = clean_drug_ingredients(INPUT_FILE, OUTPUT_FILE)

        # Display sample output
        print("\n" + "=" * 70)
        print("SAMPLE CLEANED INGREDIENTS (First 15 rows)")
        print("=" * 70)
        sample = df_result[
            ['Graph_Node_Ingredient', 'ingredient_count', 'combo_type', 'row_flag']
        ].head(15)
        for idx, row in sample.iterrows():
            ingredient = row['Graph_Node_Ingredient']
            count      = row['ingredient_count']
            combo      = row['combo_type']
            flag       = row['row_flag'] or ''
            display    = ingredient if len(ingredient) <= 60 else ingredient[:57] + "..."
            flag_str   = f" [{flag}]" if flag else ""
            print(f"{idx + 1:2d}. [{combo:6s}] ({count} ing){flag_str} {display}")

        print("=" * 70)
        print(f"\nFull results saved to : {OUTPUT_FILE}")
        print(f"Detailed log saved to : {LOG_FILE}")
        print(f"\nSTATISTICS SUMMARY:")
        print(f"   Total valid drugs : {len(df_result):,}")
        print(f"   Single ingredient : {(df_result['combo_type'] == 'single').sum():,}")
        print(f"   Combinations      : {(df_result['combo_type'] == 'combo').sum():,}")

    except Exception as e:
        log_message(f"\nERROR OCCURRED: {e}")
        import traceback
        log_message(traceback.format_exc())
        print(f"\nError: {e}")
        print("Check log file for details.")
