"""Streamlit interface for Scarlet Dish."""
from base64 import b64encode
from functools import partial
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
mode = st.radio("Find a recipe", tuple(MODE_GUIDES), horizontal=True,
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


def listing(results, pantry=False, *, heading=None):
    if heading:
        st.header(heading, anchor=False)
    if not results:
        st.info("No matching recipes. Try another query or ingredient.")
    for row in results:
        with st.expander(f"{row['title']} · {row['cuisine']} · {row['recipe_id']}"):
            st.write(row["category"])
            st.write("Matches · " + (", ".join(row.get("matched_ingredients", [])) or "None"))
            if pantry:
                st.write("Missing essentials · " + (", ".join(row["missing_essentials"]) or "None"))
                st.write("Seasonal ingredients · " + (", ".join(row["in_season_ingredients"]) or "None"))


def show_adaptation(result):
    if result["status"] != "ok":
        st.info(result["message"])
        return
    st.subheader(result["title"], anchor=False)
    st.caption("Adaptation plan. Use the original recipe's quantities.")
    if result["catalog_note"]:
        st.warning(result["catalog_note"])
    for change in result["changes"]:
        st.write(change["instruction"])
        st.caption(change["reason"])
    if result["identity_preserved"]:
        st.markdown("**What to keep**")
        for detail in result["identity_preserved"]:
            st.text(detail)
    with st.expander("Ingredients and adapted method"):
        ingredients_column, method_column = st.columns((1, 2), gap="medium")
        with ingredients_column:
            st.markdown("**Ingredients**")
            for item in result["ingredients"]:
                suffix = " (optional addition)" if item["kind"] == "optional_addition" else ""
                st.text(item["name"] + suffix)
        with method_column:
            st.markdown("**Method**")
            for step in result["steps"]:
                st.text(f"{step['position']}. {step['instruction']}")
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
