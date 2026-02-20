# 💊 DataDose  
### Pharmaceutical Data Cleaning & Normalization Engine for Drug Interaction Readiness

---

## 📌 Overview

**DataDose** is a rule-based pharmaceutical data cleaning and normalization pipeline designed to transform raw drug ingredient data into a structured, validated, graph-ready dataset.

The system prepares medication records for downstream **drug interaction detection**, ensuring data integrity, deterministic processing, and healthcare-safe normalization.

Built using:
- Python
- Pandas
- Regex-driven rule engine
- Deterministic validation architecture

---

## 🎯 Problem Statement

Raw pharmaceutical datasets are highly inconsistent and contain:

| Issue Type | Example |
|------------|----------|
| Dosage contamination | `500mg`, `100 IU/ml`, `0.5%` |
| Embedded strength | `collagen7000mg` |
| Cosmetic entries | `cream + hair + styling` |
| Corrupted tokens | `__ING0035__2` |
| Misspellings | `cholorohexidine` |
| Vague categories | `minerals`, `vitamins` |
| Inconsistent separators | `-`, `/`, `and`, `with`, `,` |
| Supplement noise | `royal jelly`, `ginseng` |

These inconsistencies make reliable drug interaction detection impossible without a strict normalization pipeline.

---

## ⚙️ Data Engineering Pipeline

| Stage | Rule ID | Description |
|-------|--------|------------|
| Token Decoding | R0 | Repair corrupted encoded placeholders (`__INGxxxx__`) |
| Spell Correction | R4 | Fix true typos only (NO synonym merging allowed) |
| Separator Normalization | R1 | Convert all separators into ` + ` |
| Dosage Removal | R3 | Remove mg, g, IU, %, embedded strengths |
| Leading Number Removal | R3.1 | Remove numeric-only leading tokens |
| Cosmetic Filtering | R5 | Delete cosmetic / personal-care rows |
| Vague Category Detection | R6 | Flag generic non-specific categories |
| Truncated Detection | R7 | Flag incomplete ingredient tokens |
| Unknown Token Detection | R8 | Flag suspicious unrecognized tokens |
| Omega Normalization | R11 | Normalize omega-3-6-9 formats |
| Intra-row Deduplication | P0.2 | Remove duplicates inside a single row only |

---

## 🧠 Immutable Data Integrity Principles

| Principle | Description |
|-----------|-------------|
| P0.1 | Synonym merging is strictly forbidden (`paracetamol ≠ acetaminophen`) |
| P0.2 | Duplicate rows allowed across dataset (only intra-row deduplication) |
| Deterministic | Same input always produces same output |
| Rule-based | No ML black-box decisions |

---

## 🔄 Before vs After Example

### Example 1 — Combination Drug

| Raw Input |
|-----------|
| `paracetamol(acetaminophen)` |

| Cleaned Output | Count | Type |
|----------------|-------|------|
| `acetaminophen + paracetamol` | 2 | combo |

---

### Example 2 — Cosmetic Entry (Deleted)

| Raw Input |
|-----------|
| `cream + hair + smooth + styling` |

| Output |
|--------|
| ❌ Row Removed (COSMETIC) |

---

### Example 3 — Normalized Combination

| Raw Input |
|-----------|
| `adenosine triphosphate+cocarboxylase+nicotinamide+vitamin b12` |

| Cleaned Output |
|----------------|
| `adenosine triphosphate + cobalamin + cocarboxylase + nicotinamide` |

---

## 🏗️ Output Schema

| Column | Description |
|--------|------------|
| `Graph_Node_Ingredient` | Normalized ingredient string (graph-ready) |
| `ingredient_count` | Number of ingredients in drug |
| `is_combination` | Boolean flag |
| `combo_type` | `single` or `combo` |

Only fully validated, flag-free rows are retained in final dataset.

---

## 📊 Cleaning Statistics (Example)

- Dataset Size: ~7MB
- Valid Drugs Retained: Automatically computed
- Combination Drugs: Automatically detected
- Single Drugs: Automatically detected
- Maximum Ingredients in Combo: Dynamic

---

## 🧪 Built-in Sanity Testing

Internal test suite validates:

- Omega normalization
- Vitamin expansion (B12 → cobalamin)
- Encoded token repair
- Spell correction
- Cosmetic deletion
- Strict synonym separation

Pipeline execution halts if sanity checks fail.

---

## 🛠️ How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the cleaning pipeline
python src/cleaning_pipeline.py
```

---

## 📁 Project Structure

```
DataDose/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   │   └── DataDoseDataset.csv
│   └── processed/
│       └── DataDoseDataset_CleanedDrugs.csv
│
├── logs/
│   └── cleaning_logFinal.txt
│
├── notebooks/
│   └── FinalDataDose.ipynb
│
└── src/
    └── cleaning_pipeline.py
```