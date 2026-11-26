"""Public Scarlet Dish inference API."""
from functools import lru_cache, partial
from pathlib import Path

from assets import configured_assets, verify_assets

RETRIEVAL_FILES = ("catalog.jsonl", "ingredient_aliases.json", "seasons.json",
                   "retrieval_config.json", "pantry_config.json")
ENCODER_FILES = ("encoder/config.json", "encoder/model.safetensors",
                 "encoder/tokenizer.json", "encoder/tokenizer_config.json")


class RecipeSystem:
    def __init__(self, artifact_dir=None, *, resolve=None):
        if resolve is not None:
            self._resolve = resolve
        elif artifact_dir:
            self._resolve = partial(verify_assets, Path(artifact_dir))
        else:
            self._resolve = configured_assets
        self._directories = {}
        self._retrieval = None
        self._adapter = None

    def _assets(self, files):
        if files not in self._directories:
            self._directories[files] = self._resolve(files=files)
        return self._directories[files]

    @property
    def retrieval(self):
        if self._retrieval is None:
            from retrieval import Retrieval

            directory = self._assets(RETRIEVAL_FILES)
            self._retrieval = Retrieval(directory, load_indexes=False, encoder_directory=self._encoder_directory)
        return self._retrieval

    def _encoder_directory(self):
        directory = self._assets(ENCODER_FILES)
        nested = directory / "encoder"
        return nested if (nested / "config.json").is_file() else directory

    def recommend_from_pantry(self, ingredients, season=None, top_k=5):
        return self.retrieval.recommend(ingredients, season, top_k)

    def find_similar_recipes(self, query, top_k=5):
        retrieval = self.retrieval
        if "fine" not in retrieval.embedding:
            directory = self._assets(("recipe_embeddings.npz",))
            retrieval.load_index("fine", directory / "recipe_embeddings.npz")
        return retrieval.find(query, top_k)

    def adapt_recipe(self, dish_name, target_cuisine="Japanese", **custom):
        """Accept a catalog name or ID, a starter name, or a recipe from any cuisine."""
        if self._adapter is None:
            from adaptation_v3 import Adapter

            self._adapter = Adapter(self._assets)
        return self._adapter.generate(dish_name, target_cuisine, **custom)


@lru_cache(maxsize=1)
def _system():
    return RecipeSystem()


def recommend_from_pantry(ingredients, season=None, top_k=5):
    return _system().recommend_from_pantry(ingredients, season, top_k)


def find_similar_recipes(query, top_k=5):
    return _system().find_similar_recipes(query, top_k)


def adapt_recipe(dish_name, target_cuisine="Japanese", **custom):
    return _system().adapt_recipe(dish_name, target_cuisine, **custom)

