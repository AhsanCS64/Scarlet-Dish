"""Construct auditable recipe rows from the five Kaggle tables."""
from collections import Counter, defaultdict
from fractions import Fraction
from hashlib import sha256
from itertools import combinations
from pathlib import Path
import json
import math
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from recipe_names import family_key, key, title_key

SEED = 126
ROOT = Path('/kaggle/input/datasets/ahsanneural/10k-south-asian-recipes-with-nutrition-and-steps')
OUT = Path('/kaggle/working/recipe_system')
FILES = {
    'recipes_master': ('recipe header and metadata', ['recipe_id']),
    'recipe_ingredients': ('ordered ingredients', ['recipe_id', 'ingredient_number']),
    'recipe_steps': ('ordered instructions', ['recipe_id', 'step_number']),
    'recipe_nutrition': ('nutrition with undocumented basis', ['recipe_id']),
    'cuisine_metadata': ('cuisine-level metadata', ['cuisine']),
}
ALIASES = {
    'onion': ['পেঁয়াজ', 'peyaj', 'piyaz', 'onions'],
    'garlic': ['রসুন', 'roshun', 'rosun', 'garlic cloves'],
    'ginger': ['আদা', 'ada', 'adrak'],
    'potato': ['আলু', 'alu', 'aloo', 'potatoes'],
    'tomato': ['টমেটো', 'tometo', 'tomatoes'],
    'spinach': ['পালং শাক', 'palong shak', 'palak'],
    'eggplant': ['বেগুন', 'begun', 'brinjal', 'aubergine'],
    'cauliflower': ['ফুলকপি', 'phulkopi', 'fulkopi', 'gobi'],
    'carrot': ['গাজর', 'gajor', 'carrots'],
    'salt': ['লবণ', 'lobon', 'namak'],
    'sugar': ['চিনি', 'chini'],
    'water': ['পানি', 'pani'],
    'oil': ['তেল', 'tel', 'cooking oil', 'vegetable oil'],
    'ghee': ['ঘি', 'ghi', 'clarified butter'],
    'basmati rice': ['বাসমতি চাল', 'basmati chal', 'basmati'],
    'rice': ['চাল', 'chal', 'chaal', 'uncooked rice'],
    'mutton': ['খাসির মাংস', 'khashir mangsho', 'khashi', 'goat meat'],
    'chicken': ['মুরগি', 'murgi', 'murghi'],
    'beef': ['গরুর মাংস', 'gorur mangsho'],
    'fish': ['মাছ', 'mach', 'maach'],
    'shrimp': ['চিংড়ি', 'chingri', 'prawn', 'prawns'],
    'hilsa': ['ইলিশ', 'ilish'],
    'lentils': ['ডাল', 'dal', 'daal', 'lentils dal', 'lentil'],
    'red lentils': ['মসুর ডাল', 'masoor dal', 'moshur dal'],
    'chickpeas': ['ছোলা', 'chola', 'chana', 'chickpea'],
    'chickpea flour': ['বেসন', 'besan', 'gram flour'],
    'all purpose flour': ['ময়দা', 'moyda', 'maida', 'plain flour'],
    'wheat flour': ['আটা', 'ata', 'atta'],
    'eggs': ['ডিম', 'dim', 'egg'],
    'yogurt': ['দই', 'doi', 'dahi', 'yoghurt', 'plain yogurt'],
    'milk': ['দুধ', 'dudh'],
    'coconut milk': ['নারকেলের দুধ', 'narkeler dudh'],
    'fresh coriander': ['ধনেপাতা', 'dhone pata', 'cilantro', 'coriander leaves'],
    'coriander powder': ['ধনে গুঁড়া', 'dhone gura', 'ground coriander'],
    'green chilli': ['কাঁচা মরিচ', 'kacha morich', 'green chili', 'green chilies', 'green chillies', 'green chilli peppers'],
    'chilli powder': ['মরিচ গুঁড়া', 'morich gura', 'chili powder', 'red chili powder', 'red chilli powder'],
    'dried red chilli': ['শুকনা মরিচ', 'shukna morich', 'dry red chili'],
    'turmeric': ['হলুদ', 'holud', 'turmeric powder', 'haldi'],
    'cumin': ['জিরা', 'jira', 'jeera', 'cumin seeds'],
    'cumin powder': ['জিরা গুঁড়া', 'ground cumin'],
    'cardamom': ['এলাচ', 'elach'],
    'cinnamon': ['দারুচিনি', 'daruchini'],
    'mustard seeds': ['সরিষা', 'shorisha', 'mustard seed'],
    'mustard oil': ['সরিষার তেল', 'shorishar tel'],
    'bottle gourd': ['লাউ', 'lau', 'lauki'],
    'pumpkin': ['মিষ্টি কুমড়া', 'mishti kumra'],
    'mango': ['আম', 'aam', 'mangoes'],
    'cucumber': ['শসা', 'shosha'],
    'green beans': ['শিম', 'shim'],
}
STAPLES = frozenset(['water', 'salt', 'oil', 'onion', 'garlic', 'ginger', 'sugar'])
SEASONS = {
    'summer': {'aliases': ['grishma', 'grisma', 'গ্রীষ্ম'], 'produce': ['mango', 'cucumber', 'bottle gourd']},
    'monsoon': {'aliases': ['barsha', 'barsa', 'বর্ষা', 'rainy'], 'produce': ['green chilli', 'bottle gourd', 'pumpkin']},
    'autumn': {'aliases': ['sharat', 'sarat', 'শরৎ'], 'produce': ['pumpkin', 'eggplant']},
    'late autumn': {'aliases': ['hemanta', 'হেমন্ত'], 'produce': ['pumpkin', 'fresh coriander', 'potato']},
    'winter': {'aliases': ['sheet', 'shit', 'shhit', 'শীত'], 'produce': ['cauliflower', 'carrot', 'spinach', 'green beans', 'tomato']},
    'spring': {'aliases': ['basanta', 'boshonto', 'বসন্ত'], 'produce': ['cucumber', 'tomato', 'eggplant']},
}
SEASON_DATA = {
    'status': 'provisional produce associations; locality and month coverage unverified',
    'season_names_source': 'https://en.banglapedia.org/index.php/Season',
    'seasons': SEASONS, 'boost': .045,
}

def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, default=str), encoding='utf-8')

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

class IngredientNormalizer:
    def __init__(self, aliases, known=()):
        self.mapping = {key(name): name for name in known}
        for canonical, variants in aliases.items():
            self.mapping.update({key(name): canonical for name in [canonical, *variants]})

    def __call__(self, raw):
        normalized = key(raw)
        return self.mapping.get(normalized, normalized)

    def recognized(self, raw):
        return key(raw) in self.mapping

QUANTITY = re.compile(r'^(\d+\s+\d+/\d+|\d+/\d+|\d+(?:\.\d+)?)\s*([a-z]+)?$', re.I)
UNITS = {
    'g': 'g', 'kg': 'kg', 'mg': 'mg', 'ml': 'ml', 'l': 'l',
    'tbsp': 'tbsp', 'tsp': 'tsp', 'cup': 'cup', 'cups': 'cup',
    'clove': 'clove', 'cloves': 'clove', 'piece': 'piece', 'pieces': 'piece',
    'medium': 'medium', 'small': 'small', 'large': 'large', 'pinch': 'pinch',
}

def quantity(raw):
    match = QUANTITY.fullmatch(raw.strip())
    if not match:
        return None, None
    number, unit = match.groups()
    if unit and unit.lower() not in UNITS:
        return None, None
    try:
        amount = sum(float(Fraction(part)) for part in number.split())
        return amount, UNITS[unit.lower()] if unit else 'count'
    except (ValueError, ZeroDivisionError):
        return None, None

def ingredient_view(ingredients):
    return ' ; '.join(ingredients)

def step_view(steps, title=''):
    text = ' '.join(steps)
    if title:
        text = re.sub(r'(?<!\w)' + re.escape(title) + r'(?!\w)', '', text, flags=re.I)
    return text

def duplicate_groups(records):
    # Compare candidate pairs within a canonical title only. Family assignment handles broader variants.
    by_title = defaultdict(list)
    for r in records:
        by_title[r['title_key']].append(r)
    sampled, pairs = [], 0
    for rows in by_title.values():
        assigned = {}
        ingredients = {r['recipe_id']: set(r['normalized_ingredients']) - STAPLES for r in rows}
        for r in rows:
            a = r['recipe_id']
            match = None
            for representative in assigned:
                left, right = ingredients[a], ingredients[representative]
                if min(len(left), len(right)) < 2:
                    continue
                overlap = len(left & right) / len(left | right)
                if overlap >= .88:
                    pairs += 1
                    if len(sampled) < 50:
                        sampled.append({'title': r['title'], 'left_id': a, 'right_id': representative,
                                        'left': sorted(left), 'right': sorted(right), 'jaccard': overlap,
                                        'left_first_step': r['steps'][0]['text'],
                                        'right_first_step': assigned[representative]['steps'][0]['text']})
                    match = representative
                    break
            r['duplicate_group'] = match or a
            if match is None:
                assigned[a] = r
    return pairs, sampled

def build_catalog(root=ROOT, out=OUT):
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    file_paths = {str(p.relative_to(root)): p for p in sorted(root.rglob('*.csv'))}
    tables, audit = {}, {'files': {}, 'joins': {}, 'limitations': []}
    for relative, path in file_paths.items():
        table = pd.read_csv(path)
        role, primary = FILES.get(path.stem, ('unmapped', []))
        audit['files'][relative] = {
            'role': role, 'rows': len(table), 'columns': list(table),
            'nulls': table.isna().sum().to_dict(), 'duplicate_rate': float(table.duplicated().mean()),
            'key': primary, 'key_duplicate_count': int(table.duplicated(primary).sum()) if primary else None,
            'key_null_count': int(table[primary].isna().any(axis=1).sum()) if primary else None,
            'sha256': sha256(path.read_bytes()).hexdigest(),
            'examples': table.head(2).to_dict('records'),
        }
        if primary:
            if path.stem in tables:
                raise ValueError(f'Multiple tables named {path.stem}')
            if table.duplicated(primary).any() or table[primary].isna().any().any():
                raise ValueError(f'Invalid key in {relative}')
            tables[path.stem] = table
    missing = set(FILES) - tables.keys()
    if missing:
        raise FileNotFoundError(f'Missing CSV tables: {sorted(missing)}')
    master, ingredients, steps, nutrition, cuisine = (tables[name] for name in FILES)
    ids = set(master.recipe_id)
    for name, table in [('ingredients', ingredients), ('steps', steps), ('nutrition', nutrition)]:
        observed = set(table.recipe_id)
        counts = table.groupby('recipe_id').size()
        audit['joins'][name] = {'keys_matched': len(ids & observed), 'missing': sorted(ids - observed),
                               'orphans': sorted(observed - ids), 'min_rows_per_key': int(counts.min()),
                               'max_rows_per_key': int(counts.max())}
    audit['joins']['cuisine'] = {'missing': sorted(set(master.cuisine)-set(cuisine.cuisine)),
                                 'unused': sorted(set(cuisine.cuisine)-set(master.cuisine)),
                                 'metadata_counts': cuisine.set_index('cuisine').recipe_count.to_dict(),
                                 'observed_counts': master.cuisine.value_counts().to_dict(),
                                 'cardinality': 'many to one'}
    explicit_normalizer = IngredientNormalizer(ALIASES)
    known = sorted({explicit_normalizer(x) for x in ingredients.ingredient_name})
    normalizer = IngredientNormalizer(ALIASES, known)
    ingredient_rows = defaultdict(list)
    for row in ingredients.sort_values(['recipe_id', 'ingredient_number']).itertuples(index=False):
        amount, unit = quantity(row.quantity)
        ingredient_rows[row.recipe_id].append({'position': row.ingredient_number, 'raw': row.ingredient_name,
            'name': normalizer(row.ingredient_name), 'quantity_raw': row.quantity, 'amount': amount, 'unit': unit,
            'category': row.category, 'explicit_alias': explicit_normalizer.recognized(row.ingredient_name)})
    step_rows = defaultdict(list)
    for row in steps.sort_values(['recipe_id', 'step_number']).itertuples(index=False):
        step_rows[row.recipe_id].append({'position': row.step_number, 'text': row.step_description,
                                          'estimated_time_minutes': row.estimated_time_minutes})
    metadata = cuisine.set_index('cuisine').to_dict('index')
    nutrition_rows = nutrition.set_index('recipe_id').to_dict('index')
    records = []
    for row in master.sort_values('recipe_id').to_dict('records'):
        rid, title = row['recipe_id'], row['recipe_name']
        ing, ordered = ingredient_rows[rid], step_rows[rid]
        names = list(dict.fromkeys(x['name'] for x in ing))
        core = [name for name in names if name not in STAPLES and
                any(x['name'] == name and x['category'] in {'Proteins', 'Vegetables', 'Grains'} for x in ing)]
        if not core:
            core = [name for name in names if name not in STAPLES]
        records.append(dict(recipe_id=rid, title=title, title_key=title_key(title), family=family_key(title),
                            cuisine=row['cuisine'], category=row['category'], servings=row['servings'],
                            metadata=row, cuisine_metadata=metadata.get(row['cuisine']), ingredients=ing,
                            normalized_ingredients=names, core=core, steps=ordered,
                            nutrition=nutrition_rows.get(rid), nutrition_basis='undocumented',
                            provenance={'dataset': root.name, 'recipe_id': rid,
                                        'recipe_author_or_url': None, 'recipe_url': None}))
    pairs, sample = duplicate_groups([r for r in records if r['steps']])
    for r in records:
        r.setdefault('duplicate_group', r['recipe_id'])
    families = np.array([r['family'] for r in records])
    entries = np.arange(len(records))
    train, heldout = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=SEED).split(entries, groups=families))
    val, test = next(GroupShuffleSplit(n_splits=1, test_size=.5, random_state=SEED).split(heldout, groups=families[heldout]))
    for split, indices in [('train', train), ('validation', heldout[val]), ('test', heldout[test])]:
        for idx in indices:
            records[idx]['split'] = split
    for field in ('family', 'duplicate_group'):
        assert pd.DataFrame(records).groupby(field).split.nunique().max() == 1
    split_rows = [{k: r[k] for k in ('recipe_id', 'title', 'family', 'duplicate_group', 'split')} for r in records]
    pd.DataFrame(split_rows).to_csv(out/'splits.csv', index=False)
    write_json(out/'ingredient_aliases.json', {'aliases': ALIASES, 'known_dataset_names': known})
    write_json(out/'seasons.json', SEASON_DATA)
    with (out/'catalog.jsonl').open('w', encoding='utf-8') as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False, default=str) + '\n')
    unique_steps = steps.step_description.value_counts()
    kacchi = [r for r in records if r['family'] == 'kacchi biryani']
    audit.update({
        'records': len(records), 'unique_ingredient_names': len(known), 'unique_exact_steps': int(unique_steps.size),
        'top_steps': unique_steps.head(15).to_dict(), 'split_counts': dict(Counter(x['split'] for x in records)),
        'grouping': {'families': len(set(families)), 'near_duplicate_links': pairs, 'samples': sample,
                     'rule': 'same normalized title, core/staple-excluded ingredient Jaccard >= .88 against a representative'},
        'quality_checks': {
            'unparsed_quantities': sum(x['amount'] is None for r in records for x in r['ingredients']),
            'vocabulary_without_curated_alias': sorted({x['name'] for r in records for x in r['ingredients'] if not x['explicit_alias']}),
            'kacchi_family_rows': len(kacchi),
            'kacchi_rows_with_mutton_and_basmati': sum({'mutton','basmati rice'} <= set(r['normalized_ingredients']) for r in kacchi),
            'kacchi_rows_with_layer_term': sum(bool(re.search(r'\blayer\w*\b', step_view([s['text'] for s in r['steps']]), re.I)) for r in kacchi),
            'kacchi_examples': [{k: r[k] for k in ('recipe_id','title','normalized_ingredients','steps')} for r in kacchi[:4]],
        },
        'limitations': ['Dish titles, ingredient lists and directions can conflict. Generic directions recur heavily.',
                        'Nutrition basis and recipe-level authorship are undocumented. Quantities are preserved even when implausible.',
                        'Near-duplicate proposals are ingredient based. Technique and authenticity need review.',
                        'Catalog recipe directions and dietary flags are unverified.'],
    })
    write_json(out/'audit.json', audit)
    return records, audit

if __name__ == '__main__':
    records, audit = build_catalog()
    print(json.dumps({k:audit[k] for k in ('records','unique_ingredient_names','unique_exact_steps','split_counts')}, indent=2))
    print({k:v for k,v in audit['quality_checks'].items() if k != 'kacchi_examples'})
