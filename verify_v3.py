"""Verify all public APIs against exported files in a fresh CPU process."""
import json
from pathlib import Path

from recipe_system import adapt_recipe, find_similar_recipes, recommend_from_pantry

RATATOUILLE = {
    "title": "Ratatouille",
    "ingredients": ["eggplant", "zucchini", "tomatoes", "bell peppers", "olive oil"],
    "steps": ["Sauté the vegetables in olive oil.", "Add tomatoes and simmer until the vegetables are tender."],
}


def test():
    pantry = {}
    for tag, ingredients, season in [
        ("winter", ["ফুলকপি", "আলু", "dal", "গাজর", "roshun"], "শীত"),
        ("monsoon", ["চিংড়ি", "pumpkin", "কাঁচা মরিচ", "coconut milk", "illusory spice"], "বর্ষা"),
    ]:
        result = recommend_from_pantry(ingredients, season)
        assert result["season"] == tag and len(result["results"]) == 5, result
        assert any(row["seasonal_boost"] > 0 for row in result["results"]), result
        assert all(set(row["matched_ingredients"]).isdisjoint(row["missing_essentials"]) for row in result["results"])
        pantry[tag] = result
    assert pantry["monsoon"]["unresolved_ingredients"]
    assert not recommend_from_pantry(["illusory spice"])["results"]

    similarity = {
        "catalog_name": find_similar_recipes("Dal Tadka"),
        "pasted": find_similar_recipes({"title": "Kacchi Biryani", "ingredients": ["mutton", "basmati rice", "ghee", "potato"],
                                      "steps": ["Marinate the mutton. Layer it under the rice. Cover and cook over low heat."]}),
        "international_paste": find_similar_recipes(RATATOUILLE),
    }
    for result in similarity.values():
        assert result["status"] == "ok" and len(result["results"]) == 5, result
        assert len({row["family"] for row in result["results"]}) == 5
        if result["source"]:
            assert all(row["family"] != result["source"]["family"] for row in result["results"])

    cases = {
        "bangladeshi_documented": ("Kacchi Biryani", "documented_source"),
        "indian_catalog": ("Dal Tadka", "catalog"),
        "pakistani_catalog": ("Karahi", "catalog"),
        "catalog_id": ("RCP00001", "catalog"),
        "north_african_name": ("Shakshuka", "starter_recipe"),
        "international_alias": ("Mac and cheese", "starter_recipe"),
        "french_paste": (RATATOUILLE, "pasted_recipe"),
        "sweet_paste": ("Chocolate cake\nIngredients:\nflour\ncocoa\nsugar\nSteps:\nMix the batter and bake in a cake tin.", "pasted_recipe"),
    }
    adaptations = {}
    for key, (query, provenance) in cases.items():
        result = adapt_recipe(query)
        assert result["status"] == "ok", result
        assert result["source"]["origin"] == provenance, result
        assert result["ingredients"] and len(result["changes"]) == 1
        assert [step["position"] for step in result["steps"]] == list(range(1, len(result["steps"]) + 1))
        assert sum(step["kind"] == "adaptation" for step in result["steps"]) == 1
        adaptations[key] = result
    assert adaptations["bangladeshi_documented"]["strategy"] == "layered_rice"
    assert adaptations["bangladeshi_documented"]["steps"][-2]["kind"] == "adaptation"
    assert "BD-OLD-DHAKA-KACCHI" in adaptations["bangladeshi_documented"]["reference_ids"]
    assert adaptations["catalog_id"]["source"]["recipe_id"] == "RCP00001"
    assert adaptations["sweet_paste"]["strategy"] == "sweet"
    french = adaptations["french_paste"]
    assert french["strategy"] == "simmered"
    assert [s["instruction"] for s in french["steps"] if s["kind"] == "source"] == RATATOUILLE["steps"]
    assert [i["name"] for i in french["ingredients"] if i["kind"] == "source"] == RATATOUILLE["ingredients"]
    for query, target, expected in [
        ("Zyphora Fantasy Dish", "Japanese", "needs_recipe"),
        ({"title": "Incomplete", "ingredients": ["lentils"]}, "Japanese", "needs_recipe"),
        ({"ingredients": [9], "steps": ["Cook."]}, "Japanese", "invalid_input"),
        ("Dal Tadka", "French", "unsupported_target"),
    ]:
        assert adapt_recipe(query, target)["status"] == expected

    return {"pantry": pantry, "similarity": similarity, "adaptations": adaptations,
            "validation": {"scope": "Automated functional verification in a fresh CPU process. Culinary quality was not taste-tested.",
                           "seasonal_pantry_cases_passed": 2, "empty_pantry_case_passed": True,
                           "similarity_cases_passed": 3, "adaptation_cases_passed": len(cases),
                           "adaptation_error_cases_passed": 4}}


if __name__ == "__main__":
    results = test()
    Path(__file__).with_name("verification_v3.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results["validation"], ensure_ascii=False), flush=True)
