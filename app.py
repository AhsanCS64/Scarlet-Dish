"""Streamlit interface for Scarlet Dish."""
from base64 import b64encode
from functools import partial
from html import escape
import os
from pathlib import Path
from threading import RLock

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from assets import AssetError, resolve_assets
from adaptation_v3 import CUSTOM_CUISINE, TARGETS

LOGO = Path(__file__).with_name("logo.png")

st.set_page_config(page_title="Scarlet Dish", page_icon=str(LOGO), layout="wide")
st.html(Path(__file__).with_name("ui.css"))
logo_uri = b64encode(LOGO.read_bytes()).decode("ascii")
st.html(f"""
<header class="scarlet-hero">
  <div>
    <p class="scarlet-kicker">South Asian recipe discovery</p>
    <h1>Scarlet Dish</h1>
    <p>Find recipes for what you have, discover dishes with similar flavors,
       or adapt a recipe to another cuisine.</p>
  </div>
  <div class="scarlet-emblem" aria-hidden="true">
    <img src="data:image/png;base64,{logo_uri}" alt="">
  </div>
</header>
""")

PANTRY_EXAMPLE = "Cauliflower, potato, lentils, carrot, garlic"
PASTE_EXAMPLE = """Ratatouille
Ingredients:
1 eggplant
2 zucchini
3 tomatoes
2 tbsp olive oil
Steps:
Sauté the vegetables in olive oil.
Add tomatoes and simmer until tender."""

MODE_GUIDES = {
    "Pantry": {
        "heading": "Cook with what you have",
        "intro": "List the ingredients you have and choose a Bangladesh season if you want seasonal suggestions.",
        "input": "Separate ingredient names with commas or put each on its own line. English and common Bengali names work.",
        "method": "Scarlet Dish ranks recipes by how much of the ingredient list you have, especially the essentials. The season gives a small boost to dishes that use seasonal produce.",
        "example": PANTRY_EXAMPLE,
        "submit": "Find pantry recipes",
        "spinner": "Matching your ingredients…",
    },
    "Similar dishes": {
        "heading": "Find similar dishes",
        "intro": "Search by dish name or recipe ID, or paste a recipe to compare it with the catalog.",
        "input": "Try Dal Tadka or RCP00001. For a pasted recipe, give it a title and put Ingredients and Steps on separate lines.",
        "method": "Results combine ingredient overlap with similarities in ingredients and cooking steps. The search filters out the source dish and near-duplicates.",
        "example": PASTE_EXAMPLE,
        "submit": "Find similar dishes",
        "spinner": "Comparing dishes…",
    },
    "Adapt a dish": {
        "heading": "Adapt a dish",
        "intro": "Choose a target cuisine, then enter a dish name, recipe ID, or a recipe with ingredients and steps.",
        "input": "Try Shakshuka with Japanese as the target. If a dish name is unfamiliar to the app, paste the full recipe instead.",
        "method": "The plan uses the source ingredients and cooking steps to suggest a seasoning or technique, when to apply it, and what to keep. Start from the original quantities.",
        "example": PASTE_EXAMPLE,
        "submit": "Create adaptation plan",
        "spinner": "Preparing your adaptation…",
    },
}
mode = st.radio("Find a recipe", ("Adapt a dish", "Pantry", "Similar dishes"), horizontal=True,
                key="scarlet_modes", label_visibility="collapsed")
guide = MODE_GUIDES[mode]


@st.cache_resource(show_spinner=False)
def runtime(local, repo, revision, _token):
    from recipe_system import RecipeSystem

    # Hub permissions are app-wide. Avoid putting the token in Streamlit's cache key.
    resolve = partial(resolve_assets, local, repo, revision, _token)
    return RecipeSystem(resolve=resolve), RLock()


def setting(name):
    value = os.getenv(name)
    if value is not None:
        return value
    try:
        return st.secrets.get(name)
    except StreamlitSecretNotFoundError:
        return None


def ingredient_chips(items, kind="match"):
    if not items:
        return ""
    extra = "scarlet-chip-seasonal" if kind == "seasonal" else "scarlet-chip-missing" if kind == "missing" else ""
    return "".join(f'<span class="scarlet-chip {extra}">{item}</span>' for item in items)


def listing(results, pantry=False, *, heading=None):
    if heading:
        st.header(heading, anchor=False)
    if not results:
        st.info("No matching recipes. Try another query or ingredient.")
    for row in results:
        score = row.get("score", 0)
        matched = row.get("matched_ingredients", []) or []
        with st.expander(f"{row['title']}  ·  {row['recipe_id']}"):
            st.html(
                f"""
                <div class="scarlet-result-meta">
                  <span class="scarlet-score">{score:.2f}</span>
                  <span class="scarlet-tag scarlet-cuisine">{row.get('cuisine', '')}</span>
                  <span class="scarlet-tag scarlet-category">{row.get('category', '')}</span>
                </div>
                """
            )
            if matched:
                st.html('<p class="scarlet-result-label">Matched ingredients</p>')
                st.html(ingredient_chips(matched, "match"))
            if pantry:
                seasonal = row.get("in_season_ingredients", []) or []
                missing = row.get("missing_essentials", []) or []
                if seasonal:
                    st.html('<p class="scarlet-result-label">In season</p>')
                    st.html(ingredient_chips(seasonal, "seasonal"))
                if missing:
                    st.html('<p class="scarlet-result-label">Missing essentials</p>')
                    st.html(ingredient_chips(missing, "missing"))
            if not matched and not pantry:
                st.caption("No shared ingredient tags.")


def show_adaptation(result):
    if result["status"] != "ok":
        st.info(result["message"])
        return
    st.html(f'<div class="adapt-hero"><span class="adapt-eyebrow">Your cuisine adaptation</span><h2>{escape(str(result["title"]))}</h2><p>Use the original recipe’s quantities as your starting point.</p></div>')
    if result["catalog_note"]:
        st.warning(result["catalog_note"])
    st.markdown("### The flavor plan")
    for number, change in enumerate(result["changes"], 1):
        st.html(f'<div class="adapt-change"><span class="adapt-number">{number:02d}</span><div><p>{escape(str(change["instruction"]))}</p><small>{escape(str(change["reason"]))}</small></div></div>')
    if result["identity_preserved"]:
        st.html('<div class="adapt-keep"><strong>Keep the heart of the dish</strong><div>' + "".join(f'<span>{escape(str(detail))}</span>' for detail in result["identity_preserved"]) + '</div></div>')
    ingredients_tab, method_tab = st.tabs((f'Ingredients ({len(result["ingredients"])})', f'Method ({len(result["steps"])})'))
    with ingredients_tab:
        st.caption("Use the original recipe for amounts. Optional additions are marked below.")
        st.html('<div class="adapt-ingredients">' + "".join(
            f'<div class="adapt-ingredient"><span>{escape(str(item["name"]))}</span>'
            + ('<small>Optional addition</small>' if item["kind"] == "optional_addition" else '') + '</div>'
            for item in result["ingredients"]
        ) + '</div>')
    with method_tab:
        for step in result["steps"]:
            st.html(f'<div class="adapt-step"><span>{escape(str(step["position"]))}</span><p>{escape(str(step["instruction"]))}</p></div>')
    if result["references"]:
        with st.expander("References"):
            for reference in result["references"]:
                st.link_button(reference["title"], reference["url"])


input_column, guide_column = st.columns((1.65, 1), gap="medium")

with input_column, st.container(key="scarlet_inputs"):
    st.header(guide["heading"], anchor=False)
    st.html(f'<p class="scarlet-intro">{guide["intro"]}</p>')
    if mode == "Adapt a dish":
        target = st.selectbox(
            "Target cuisine", (*TARGETS, CUSTOM_CUISINE), accept_new_options=True,
            help="Type to search. Choose Custom cuisine to provide your own ingredients or techniques.",
        )
    with st.form("recipe_search", border=False):
        if mode == "Pantry":
            query = st.text_area("Ingredients", placeholder=PANTRY_EXAMPLE, height=120)
            season_column, count_column = st.columns(2)
            seasons = ("Any season", "Summer", "Monsoon", "Autumn", "Late autumn", "Winter", "Spring")
            season = season_column.selectbox(
                "Bangladesh season", seasons, help="Choose Any season to rank without a seasonal boost.",
            )
            top_k = count_column.slider("Results", 1, 10, 5)
        else:
            query = st.text_area(
                "Dish name, recipe ID or recipe",
                placeholder="Dal Tadka or RCP00001\n\nFor a full recipe, use Ingredients and Steps headings.",
                height=146,
            )
            if mode == "Similar dishes":
                top_k = st.slider("Results", 1, 10, 5)
            elif target == CUSTOM_CUISINE:
                st.caption("Give your custom cuisine a name and fill in at least one of the two fields below.")
                custom_name = st.text_input("Custom cuisine name", placeholder="Levantine", max_chars=80)
                custom_ingredients = st.text_area(
                    "Characteristic ingredients", placeholder="Sumac, parsley, lemon", max_chars=500,
                )
                custom_techniques = st.text_area(
                    "Characteristic techniques", placeholder="Char a vegetable garnish", max_chars=500,
                )
        submitted = st.form_submit_button(guide["submit"], type="primary")

with guide_column, st.container(key="scarlet_guide"):
    st.header("How to use it", anchor=False)
    st.html(f"""
    <div class="scarlet-explanation">
      <p><strong>What to enter.</strong> {guide["input"]}</p>
      <p><strong>How it works.</strong> {guide["method"]}</p>
    </div>
    <p class="scarlet-example-label">Example input</p>
    """)
    st.code(guide["example"], language=None)

if submitted:
    with st.container(key="scarlet_results"):
        if not query.strip():
            st.info("Enter ingredients or a dish first.")
        else:
            try:
                with st.spinner(guide["spinner"]):
                    system, lock = runtime(
                        setting("RECIPE_SYSTEM_DIR"), setting("SCARLET_HF_REPO"),
                        setting("SCARLET_HF_REVISION"), setting("HF_TOKEN"),
                    )
                    with lock:
                        if mode == "Pantry":
                            result = system.recommend_from_pantry(query, None if season == "Any season" else season, top_k)
                        elif mode == "Similar dishes":
                            result = system.find_similar_recipes(query.strip(), top_k)
                        else:
                            custom = {}
                            if target == CUSTOM_CUISINE:
                                custom = {
                                    "custom_name": custom_name, "custom_ingredients": custom_ingredients,
                                    "custom_techniques": custom_techniques,
                                }
                            result = system.adapt_recipe(query, target, **custom)
                if mode == "Pantry":
                    if result["unresolved_ingredients"]:
                        st.caption("Unrecognized · " + ", ".join(result["unresolved_ingredients"]))
                    listing(result["results"], pantry=True, heading="Pantry matches")
                elif mode == "Similar dishes":
                    if result["status"] == "ambiguous":
                        st.info("Choose one of these recipe IDs to identify the source dish.")
                        listing(result["choices"])
                    else:
                        listing(result["results"], heading="Similar dishes found")
                else:
                    show_adaptation(result)
            except LookupError:
                st.info("No matching dish name. Try a recipe ID or paste the recipe.")
            except AssetError as error:
                st.error(str(error))
            except ValueError as error:
                st.info(str(error))
            except (ImportError, OSError, RuntimeError) as error:
                st.error(f"Inference failed ({type(error).__name__}). Check the installed requirements and V3 assets.")
