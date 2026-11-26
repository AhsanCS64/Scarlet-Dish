# Scarlet Dish V3

**Scarlet Dish** is a South Asian recipe discovery app with three modes, a fine-tuned sentence encoder, and a Scarlet Witch–themed UI. It runs fully offline on CPU once the local asset bundle is in place.

---

## Modes

| Mode | Input | What it does |
|---|---|---|
| **Pantry** | Comma- or newline-separated ingredients + optional Bangladesh season | Ranks recipes by how much of your pantry they need. Shows matched ingredients, missing essentials, and in-season items. |
| **Similar dishes** | Dish name, recipe ID (`RCP00001`), or pasted recipe | Combines TF-IDF ingredient Jaccard with fine-tuned MiniLM embeddings on ingredients and steps. Excludes the source family and near-duplicates. |
| **Adapt a dish** | Dish name, recipe ID, or pasted recipe + target cuisine | Generates a cuisine adaptation plan: what to change, when to add it, and what to keep. Uses original quantities. |

Supported adaptation targets: Japanese, Korean, Thai, Mexican, Italian, Turkish, Bangladeshi, **Custom cuisine** (enter your own name, ingredients, or techniques).

---

## Run locally (Windows)

**First-time setup** (from the root of a clone of this GitHub repository):

```powershell
py -3.13 -m venv .venv
$py = (Resolve-Path '.\.venv\Scripts\python.exe').Path
& $py -m pip install -r requirements.txt
& $py -m pip check
```

Configure the asset source before starting. For a standalone clone, use the Hub repo and full commit SHA from the deployment settings below (add `HF_TOKEN` only for a private Hub repo):

```powershell
$env:SCARLET_HF_REPO = "Archmage64/scarlet-dish-v3"
$env:SCARLET_HF_REVISION = "FULL_40_CHARACTER_COMMIT_SHA"
# $env:HF_TOKEN = "hf_..."  # private Hub repo only
```

From the full source workspace, use its local bundle instead:

```powershell
$env:RECIPE_SYSTEM_DIR = (Resolve-Path '..\local-assets').Path
```

Start Streamlit from the repository root with the venv Python:

```powershell
& $py -m streamlit run app.py
```

Always use the venv Python; the system Python may be missing `transformers`, which causes the *"Inference failed (ModuleNotFoundError)"* error in Similar dishes.

**Try these queries after startup:**

- Pantry → `Cauliflower, potato, lentils, carrot, garlic` with season `Winter`
- Similar dishes → `Dal Tadka`
- Adapt a dish → `Shakshuka` with target `Japanese`

---

## Runtime asset resolution

The app resolves assets in this order:

1. `RECIPE_SYSTEM_DIR` env var (or `st.secrets`) → verifies files against `assets_v3.json`
2. `SCARLET_HF_REPO` + `SCARLET_HF_REVISION` → downloads matching files from Hugging Face Hub
3. Auto-discovery: looks for `catalog.jsonl` in `github/`, `release-v3/local-assets/`, or `recipe_system_v3/`

All 11 production files are verified by byte size and SHA-256 before use. The required files are listed in `assets_v3.json`.

---

## Deployment — Streamlit Community Cloud

1. Push the contents of the `github/` folder to the root of a GitHub repository. The detailed release runbook is in `release-v3/RELEASE.md` in the source workspace; it is not part of this deployable folder.
2. In **Create app**, select `AhsanCS64/Scarlet-Dish`, branch `master`, file `app.py`, Python **3.13**.
3. Paste into **Secrets**:

```toml
SCARLET_HF_REPO     = "Archmage64/scarlet-dish-v3"
SCARLET_HF_REVISION = "FULL_40_CHARACTER_COMMIT_SHA"
# HF_TOKEN = "hf_..."   # only needed for a private Hub repo
```

The app downloads and caches only the files listed in `assets_v3.json`. The catalog, fine-tuned encoder, and recipe index must come from the same export.

> **Before publishing the Hugging Face bundle:** confirm you have redistribution rights for the source recipe text and derived artifacts. The source data card describes third-party website and cookbook sources without identifying which recipes came from them. Keep the bundle private until those rights and required attribution/ShareAlike terms are confirmed. If the Hub repo is private, configure a read-only `HF_TOKEN` in Streamlit secrets.

---

## Editing cuisine profiles

Open `cuisine_profiles.json` and restart Streamlit. Each profile has these rule keys:

| Key | Used when |
|---|---|
| `rice` | Source is a rice dish |
| `simmered` | Source is a stew / curry |
| `roasted` | Source is roasted / grilled |
| `sweet` | Source is a dessert |
| `finish` | Always appended last |
| `layered_rice` | Overrides `rice` for layered rice (e.g. biryani) |
| `wet_heat` | Overrides `simmered` for braised dishes |

Each rule has `ingredients`, `instruction`, and `reason` fields. Instructions work best when they specify *when* to add the seasoning and preserve the source method.

---

## Source workspace layout

```
release-v3/
├── github/                 ← Git-tracked app code (deploy this)
│   ├── app.py              ← Streamlit entry point
│   ├── recipe_system.py    ← Public inference API
│   ├── retrieval.py        ← Pantry + similarity engine (TF-IDF + embeddings)
│   ├── adaptation_v3.py    ← Cuisine adaptation plan generator
│   ├── encoder.py          ← MiniLM fine-tuning + encode()
│   ├── catalog.py          ← Ingredient normalizer, catalog utilities
│   ├── assets.py           ← Asset resolution and SHA-256 verification
│   ├── assets_v3.json      ← Locked file hashes for the V3 bundle
│   ├── cuisine_profiles.json ← Editable per-cuisine adaptation rules
│   ├── ui.css              ← Scarlet Witch–themed Streamlit CSS
│   ├── logo.png            ← App logo
│   ├── verify_v3.py        ← CPU verifier (runs without Streamlit)
│   ├── recipe_names.py     ← Title / family key helpers
│   └── requirements.txt    ← Pinned Python dependencies
│
├── local-assets/           ← 11 production files (not in Git)
│   ├── catalog.jsonl       ← 9,997 normalized recipes (JSONL)
│   ├── recipe_embeddings.npz ← Fine-tuned ingredient + step vectors
│   ├── encoder/            ← Fine-tuned MiniLM weights (safetensors)
│   ├── ingredient_aliases.json
│   ├── pantry_config.json
│   ├── retrieval_config.json
│   ├── seasons.json
│   └── adaptation_references.json
│
├── checks/                 ← Verification scripts and logs
│   ├── run_isolated.py     ← Runs all checks in isolation
│   ├── test_app.py         ← Streamlit AppTest cases
│   ├── check_api.py        ← API-level test cases
│   └── test_cuisines.py    ← Multi-cuisine adaptation tests
│
├── run_app.bat             ← Windows launch script (double-click)
├── run_app.ps1             ← PowerShell launch script
├── RELEASE.md              ← Full release runbook (Git + HF + Cloud)
├── TEST_RESULTS.md         ← Verification results log
└── artifact-inventory.csv  ← SHA-256 inventory of original ZIP contents
```

---

## Data and models

- **Catalog**: Adapted from Muhammad Ahsan's *South Asian Recipes with Nutrition & Steps* (Kaggle, CC BY-SA 4.0). Ingredients normalized, duplicates grouped, train/validation/test splits added. 9,997 recipes across Indian, Pakistani, Bangladeshi, Sri Lankan, and Nepali cuisines.
- **Encoder**: Fine-tuned from [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41) (Apache-2.0). Contrastive training on ingredient↔step pairs using in-batch negatives.
- **Pantry ranker**: Weighted baseline with core coverage, missing essentials, weighted TF-IDF overlap, and seasonal compatibility.

> Catalog quantities and directions can be inconsistent. Use a trusted source recipe for quantities and timing. All evaluations use automatic proxy judgments.

[Kaggle data card](https://www.kaggle.com/datasets/ahsanneural/10k-south-asian-recipes-with-nutrition-and-steps) · [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) · [Base MiniLM model](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41)
