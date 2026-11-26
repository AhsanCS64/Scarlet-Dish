"""Recipe-grounded cuisine adaptation plans."""
from collections.abc import Mapping
from pathlib import Path
import json
import re

from recipe_names import family_key, key, title_key


PROFILES_FILE = Path(__file__).with_name("cuisine_profiles.json")
CUSTOM_CUISINE = "Custom cuisine"


def load_profiles():
    with PROFILES_FILE.open(encoding="utf-8") as stream:
        return json.load(stream)


TARGETS = tuple(load_profiles())
SOURCE_PROFILES = {
    "kacchi biryani": {
        "title": "Kacchi Biryani",
        "cuisine": "Bangladeshi",
        "ingredients": ["mutton", "basmati rice", "potatoes", "onions", "ghee",
                        "plain yogurt", "ginger", "garlic", "biryani spices"],
        "steps": [
            "Marinate the raw mutton with yogurt, ginger, garlic and biryani spices. Prepare the potatoes and fry the onions.",
            "Parboil the basmati rice separately and drain it.",
            "Layer the raw marinated mutton and potatoes below the rice. Add the fried onions and ghee.",
            "Cook the layers together in a tightly covered pot over low heat. Check meat doneness before serving.",
        ],
        "identity": ["Raw marinated mutton goes in the pot first", "Distinct basmati rice and mutton layers",
                     "Tightly covered, low-heat dum cooking"],
        "reference": "BD-OLD-DHAKA-KACCHI",
    },
    "shakshuka": {
        "title": "Shakshuka",
        "cuisine": "North African",
        "ingredients": ["eggs", "tomatoes", "bell pepper", "onion", "garlic",
                        "olive oil", "cumin", "paprika", "salt"],
        "steps": [
            "Soften the onion, pepper and garlic in olive oil.",
            "Add tomatoes, cumin, paprika and salt. Simmer into a thick sauce.",
            "Make wells in the sauce, crack in the eggs and cook them covered in the same pan.",
        ],
        "identity": ["Tomato and pepper sauce", "Eggs cooked in wells in the sauce"],
    },
    "macaroni and cheese": {
        "title": "Macaroni and Cheese",
        "ingredients": ["elbow macaroni", "cheddar", "milk", "butter", "flour", "salt"],
        "steps": [
            "Boil and drain the macaroni.",
            "Whisk butter, flour and milk into a smooth sauce, then stir in cheddar.",
            "Fold the pasta into the cheese sauce and serve.",
        ],
        "identity": ["Elbow macaroni", "Cheddar sauce"],
    },
}
SOURCE_ALIASES = {"mac and cheese": "macaroni and cheese"}

def _texts(value, field):
    if isinstance(value, str):
        value = value.splitlines()
    if not isinstance(value, (tuple, list)) or any(not isinstance(v, str) for v in value):
        raise ValueError(f"{field} must be text or a list of text entries.")
    if len(value) > 100:
        raise ValueError(f"Provide at most 100 {field.lower()} entries.")
    if sum(len(v) for v in value) > 16_000:
        raise ValueError(f"{field} is too long. Keep the relevant recipe details.")
    return [v.strip().lstrip("-• ").strip() for v in value if v.strip().lstrip("-• ").strip()]


def parse_recipe(source):
    if isinstance(source, Mapping):
        title = source.get("title") or source.get("dish_name") or "Your recipe"
        if not isinstance(title, str) or len(title.strip()) > 150 or "\n" in title:
            raise ValueError("Use a dish name of at most 150 characters.")
        return title.strip() or "Your recipe", _texts(source.get("ingredients", []), "Ingredients"), _texts(source.get("steps", []), "Steps")
    lines = [line.strip() for line in source.splitlines() if line.strip()]
    data = {"ingredients": [], "steps": []}
    title, section = "Your recipe", None
    for i, line in enumerate(lines):
        clean = line.lstrip("# ").strip()
        match = re.match(r"^(ingredients?|steps?|directions|method|instructions)\s*(?::\s*(.*))?$", clean, re.I)
        if match:
            section = "ingredients" if match[1].lower().startswith("ingredient") else "steps"
            if match[2]:
                data[section].append(match[2])
        elif section:
            data[section].append(line)
        elif i == 0:
            title = clean
    return parse_recipe({"title": title, **data})


def strategy_for(ingredients, steps, title="", category=""):
    pantry = " ".join(ingredients).casefold()
    prep = " ".join(steps).casefold()
    label = f"{title} {category}".casefold()
    if re.search(r"\b(?:desserts?|cakes?|cookies?|pudding|brownies?|ice cream|sweet buns?|cheesecake)\b", label):
        return "sweet"
    if re.search(r"\b(?:basmati|rice)\b", pantry) and re.search(r"\blayer\w*\b", prep):
        return "layered_rice"
    if re.search(r"\b(?:basmati|rice)\b", pantry) and (re.search(r"\brice\b", prep) or re.search(r"\b(?:rice|biryani|pulao|pilaf)\b", label)):
        return "rice"
    if re.search(r"\b(?:broth|soups?|stock)\b", label) or re.search(r"\b(?:broth|stock)\b", pantry):
        return "wet_heat"
    if re.search(r"\b(?:simmer|stew|braise|sauces?|gravy)\w*\b", prep):
        return "simmered"
    if re.search(r"\b(?:bake|roast|grill|fry)\w*\b", prep):
        return "roasted"
    return "finish"


def _names(row):
    return [entry["raw"] for entry in row["ingredients"] if entry.get("raw")]


def _catalog_source(row):
    names = _names(row)
    return {
        "title": row["title"], "recipe_id": row["recipe_id"], "cuisine": row["cuisine"],
        "category": row["category"], "ingredients": names,
        "steps": [entry["text"] for entry in row["steps"] if entry.get("text")],
        "identity": [name for name in names if name.casefold() in row.get("core", [])],
        "origin": "catalog",
        "ingredient_quantities": [
            f"{entry['raw']} ({entry['quantity_raw']})" if entry.get("quantity_raw") else entry["raw"]
            for entry in row["ingredients"] if entry.get("raw")
        ],
    }


def custom_rule(strategy, ingredients, techniques):
    timing = {
        "rice": "After the rice is cooked, try the adjustment on one serving.",
        "layered_rice": "After the rice is cooked, try the adjustment on the top of one serving and keep the layers intact.",
        "simmered": "When the sauce has finished cooking, ladle out a small portion to test the adjustment.",
        "wet_heat": "When the broth has finished cooking, ladle out a small portion to test the adjustment.",
        "roasted": "After the original grilling or roasting step, test the adjustment on one cooked portion.",
        "sweet": "Once the dessert is baked, chilled or set, test the adjustment on one serving.",
        "finish": "Set aside one serving once the dish is cooked and test the adjustment before serving.",
    }[strategy]
    instruction = [timing]
    if ingredients:
        instruction.append(f"Use the characteristic ingredients you supplied ({ingredients}) to prepare a small topping or side. Use each ingredient in its normally prepared form and taste before adding more.")
    if techniques:
        instruction.append(f"Try the technique you supplied ({techniques}) on a separate topping, garnish, or portion if suitable. Keep the source dish’s cooking time and method.")
    return {
        "ingredients": ingredients,
        "instruction": " ".join(instruction),
        "reason": "A separate portion lets you try the flavors and techniques you supplied while preserving the original recipe.",
    }


def define_identity(source, strategy):
    if source["origin"] == "pasted_recipe":
        preserved = ["Keep the source ingredients and quantities, including " + ", ".join(source["ingredients"][:4]) + "."]
    elif source["origin"] == "catalog" and source["identity"]:
        preserved = ["Keep the catalog’s core ingredients: " + ", ".join(source["identity"]) + "."]
    else:
        preserved = source["identity"]
    cues = {
        "layered_rice": r"\blayer\w*\b",
        "rice": r"\brice\b",
        "simmered": r"\b(?:simmer|stew|braise|boil)\w*\b",
        "wet_heat": r"\b(?:broth|stock|boil|simmer)\w*\b",
        "roasted": r"\b(?:bake|roast|grill|fry)\w*\b",
        "sweet": r"\b(?:bake|chill|set|whisk|fold)\w*\b",
    }
    method = next((step for step in source["steps"] if re.search(cues.get(strategy, r"(?!)"), step, re.I)), None)
    if method:
        return [*preserved, f"Keep this source step: {method}"]
    return [*preserved, "Keep the original cooking method and seasoning."]


class Adapter:
    def __init__(self, resolve_assets):
        self.resolve_assets = resolve_assets
        self.profiles = load_profiles()

    def catalog_source(self, token, title, family):
        directory = self.resolve_assets(("catalog.jsonl",))
        family_match = None
        with (directory / "catalog.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row["recipe_id"] == token.upper() or row["title_key"] == title:
                    return row
                if family_match is None and row["family"] == family:
                    family_match = row
        return family_match

    def source(self, value):
        if isinstance(value, Mapping) or (isinstance(value, str) and (
            "\n" in value.strip() or re.match(r"^\s*(?:ingredients?|steps?|method|directions|instructions)\s*:", value, re.I)
        )):
            title, ingredients, steps = parse_recipe(value)
            if not ingredients or not steps:
                return {"status": "needs_recipe", "message": "Provide both ingredients and cooking steps for the dish you want to adapt."}
            return {"title": title, "ingredients": ingredients, "steps": steps, "origin": "pasted_recipe",
                    "identity": ["Original ingredient list and cooking method"]}
        if not isinstance(value, str) or not value.strip() or len(value.strip()) > 150:
            raise ValueError("Enter a dish name, recipe ID or a recipe with ingredients and steps.")
        token = value.strip()
        q = title_key(token)
        normalized = family_key(q)
        profile = SOURCE_PROFILES.get(SOURCE_ALIASES.get(normalized, normalized))
        if profile:
            origin = "documented_source" if profile.get("reference") else "starter_recipe"
            return dict(profile, origin=origin)
        row = self.catalog_source(token, q, normalized)
        if row is not None:
            return _catalog_source(row)
        return {
            "status": "needs_recipe",
            "message": "This dish name is outside the catalog and starter recipes. Paste its ingredients and cooking steps to adapt it.",
        }

    def generate(self, dish_name, target_cuisine, *, custom_name="", custom_ingredients="", custom_techniques=""):
        target_key = key(target_cuisine)
        target_key = target_key.removesuffix(" inspired")
        target = next((name for name in self.profiles if key(name) == target_key), None)
        custom = target_key == key(CUSTOM_CUISINE)
        if custom:
            if not all(isinstance(value, str) for value in (custom_name, custom_ingredients, custom_techniques)):
                return {"status": "invalid_input", "message": "Provide the custom cuisine details as text."}
            target = custom_name.strip()
            custom_ingredients = custom_ingredients.strip()
            custom_techniques = custom_techniques.strip()
            if not target or len(target) > 80 or max(len(custom_ingredients), len(custom_techniques)) > 500:
                return {"status": "invalid_input", "message": "Enter a cuisine name (up to 80 characters) and up to 500 characters each for ingredients and techniques."}
            if not (custom_ingredients or custom_techniques):
                return {"status": "invalid_input", "message": "Describe at least one characteristic ingredient or technique for your custom cuisine."}
        elif target is None:
            return {
                "status": "unsupported_target", "supported_targets": list(self.profiles),
                "message": "Choose Custom cuisine to use this target. Enter its name and describe characteristic ingredients or techniques.",
            }
        try:
            source = self.source(dish_name)
        except ValueError as error:
            return {"status": "invalid_input", "message": str(error)}
        if "status" in source:
            return source
        if not source["ingredients"] or not source["steps"]:
            return {"status": "needs_recipe", "message": "This catalog entry is incomplete. Paste a recipe with ingredients and steps."}
        origin = source["origin"]
        strategy = strategy_for(source["ingredients"], source["steps"], source["title"], source.get("category", ""))
        japanese = target == "Japanese" and not custom
        if japanese:
            if origin == "catalog":
                strategy = {"layered_rice": "rice", "simmered": "finish"}.get(strategy, strategy)
        elif strategy == "finish" and re.search(r"\b(?:dal|dhal|stew|karahi)\b", key(source["title"])):
            strategy = "simmered"
        if custom:
            rule = custom_rule(strategy, custom_ingredients, custom_techniques)
        else:
            profile = self.profiles[target]
            fallback = {"layered_rice": "rice", "wet_heat": "simmered"}.get(strategy, strategy)
            rule = profile.get(strategy) or profile[fallback]
        addition = rule["ingredients"]
        instruction = rule["instruction"]
        action = {"ingredient": addition, "instruction": instruction, "reason": rule["reason"]}
        ref_ids, used_references = self.references_for(rule.get("reference"), source)
        preserved = source["identity"]
        if not japanese:
            preserved = define_identity(source, strategy)
        instructions = [{"instruction": text, "kind": "source"} for text in source["steps"]]
        if strategy == "layered_rice" and japanese:
            layer = next(i for i, step in enumerate(source["steps"]) if re.search(r"\blayer\w*\b", step, re.I))
            instructions.insert(layer + 1, {"instruction": instruction, "kind": "adaptation"})
        else:
            instructions.append({"instruction": instruction, "kind": "adaptation"})
        for i, step in enumerate(instructions, 1):
            step["position"] = i
        return {
            "status": "ok",
            "title": f"{target}-inspired {source['title']}",
            "target_cuisine": target,
            "source": {k: v for k, v in source.items() if k not in {"ingredients", "steps", "ingredient_quantities"}},
            "identity_preserved": preserved,
            "ingredients": [{"name": name, "kind": "source"} for name in (
                source.get("ingredient_quantities", source["ingredients"])
                if not japanese else source["ingredients"]
            )] + ([{"name": addition, "kind": "optional_addition"}] if addition else []),
            "steps": instructions,
            "changes": [action],
            "strategy": strategy,
            "reference_ids": ref_ids,
            "references": used_references,
            "catalog_note": "Catalog quantities and steps can be inconsistent. Use a trusted recipe for quantities and cooking times." if origin == "catalog" else "",
        }

    def references_for(self, reference_id, source):
        if not (reference_id or source.get("reference")):
            return [], []
        directory = self.resolve_assets(("adaptation_references.json",))
        with (directory / "adaptation_references.json").open(encoding="utf-8") as stream:
            references = json.load(stream)
        ref_ids, used = [], []
        for ref in references["target_references"]:
            if ref["id"] == reference_id:
                ref_ids.append(reference_id)
                used.append(ref)
        for ref in references["source_technique_notes"].values():
            if ref.get("reference_id") == source.get("reference"):
                ref_ids.append(ref["reference_id"])
                used.append({k: ref[k] for k in ("title", "publisher", "url")})
        return ref_ids, used
