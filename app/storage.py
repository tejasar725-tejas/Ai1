"""Storage module for the Recipe Step Autocompleter.

Handles loading, saving, and managing recipe corpora and trained model artifacts.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional


DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_RECIPES_FILE = DEFAULT_DATA_DIR / "recipes.txt"


# Built-in curated recipe step instructions covering diverse cooking actions:
# sautéing, roasting, baking, boiling pasta, preparing sauces, soups, salads,
# seasonings, and garnishes.
BUILTIN_RECIPE_STEPS: List[str] = [
    # --- Sautéing, Frying, & Skillet Cooking ---
    "Heat olive oil in a large skillet over medium-high heat until shimmering.",
    "Heat vegetable oil in a large non-stick pan over medium heat.",
    "Heat olive oil in a deep skillet and cook the garlic until fragrant, about 1 minute.",
    "Heat butter and olive oil in a saucepan over medium-low heat until melted.",
    "Add minced garlic and diced yellow onions, sautéing until fragrant and translucent, about 3 to 4 minutes.",
    "Add chopped garlic, sliced shallots, and ginger to the hot skillet, stirring constantly for 1 minute.",
    "Melt unsalted butter in a heavy-bottomed skillet over medium heat.",
    "Melt 2 tablespoons of butter in a large pan and add the sliced mushrooms.",
    "Sear the seasoned chicken breasts in the hot skillet for 6 to 8 minutes per side until golden brown and cooked through.",
    "Sear the steak in the cast iron skillet for 4 minutes per side until a deep crust forms.",
    "Add sliced button mushrooms and cook, stirring occasionally, until browned and tender.",
    "Cook the ground beef in a skillet over medium heat, breaking it up with a wooden spoon until thoroughly browned.",
    "Cook the Italian sausage over medium heat until browned, then drain excess fat.",
    "Deglaze the skillet with dry white wine, scraping up all the browned bits from the bottom of the pan.",
    "Deglaze the pan with chicken stock, scraping up any delicious browned bits with a wooden spoon.",
    "Stir in freshly chopped spinach and cook until wilted, about 2 minutes.",
    "Add the bell peppers and zucchini to the pan and stir-fry for 5 minutes until crisp-tender.",
    "Stir-fry the chicken and vegetables in sesame oil over high heat for 6 minutes.",

    # --- Boiling, Steaming, & Pasta ---
    "Bring a large pot of salted water to a rolling boil over high heat.",
    "Bring a medium saucepan of lightly salted water to a gentle boil.",
    "Bring 4 cups of water to a boil, then stir in the white rice and a pinch of salt.",
    "Add the pasta to the boiling water and cook until al dente, stirring occasionally to prevent sticking.",
    "Add the spaghetti to boiling salted water and cook for 8 to 10 minutes until tender.",
    "Drain the cooked pasta thoroughly in a colander, reserving 1/2 cup of starchy pasta water.",
    "Drain the pasta and return it to the warm pot with a tablespoon of olive oil.",
    "Toss the drained pasta with olive oil, minced garlic, and freshly grated Parmesan cheese.",
    "Boil the peeled russet potatoes in salted water for 15 to 20 minutes until fork-tender.",
    "Drain the potatoes and return them to the warm pot to steam dry for 1 minute.",
    "Steam the broccoli florets in a steamer basket over boiling water for 5 minutes until vibrant green.",

    # --- Baking, Doughs, & Roasting ---
    "Preheat oven to 375 degrees F (190 degrees C) and grease a 9x13-inch baking dish with butter.",
    "Preheat oven to 400 degrees F (200 degrees C) and line a large baking sheet with parchment paper.",
    "Preheat oven to 350 degrees F (175 degrees C) and grease and flour two 9-inch round cake pans.",
    "Preheat the oven to 425 degrees F (220 degrees C) for roasting vegetables.",
    "Whisk together the all-purpose flour, baking powder, baking soda, and kosher salt in a large bowl.",
    "In a medium bowl, whisk together the flour, cocoa powder, cornstarch, and fine sea salt.",
    "Beat the softened butter and granulated sugar together until light and fluffy, about 3 minutes.",
    "Beat the eggs and sugar in a stand mixer on medium speed until pale and thick.",
    "Add the eggs one at a time, beating well after each addition, then stir in pure vanilla extract.",
    "Gradually fold the dry ingredients into the wet mixture until just combined, being careful not to overmix.",
    "Pour the batter evenly into the prepared baking pans and smooth the top with a silicone spatula.",
    "Bake in the preheated oven for 25 to 30 minutes, or until a toothpick inserted into the center comes out clean.",
    "Bake in the preheated oven for 12 to 15 minutes until the cookies are golden around the edges.",
    "Transfer the cake pans to a wire cooling rack and allow to cool completely for 15 minutes before frosting.",
    "Knead the bread dough on a lightly floured surface for 8 to 10 minutes until smooth and elastic.",
    "Place the dough in a lightly oiled bowl, cover with plastic wrap, and let rise in a warm spot until doubled in size.",
    "Roll out the chilled pie crust on a lightly floured work surface to a 12-inch circle.",
    "Toss the chopped root vegetables with olive oil, kosher salt, black pepper, and fresh rosemary sprigs.",
    "Spread the vegetables in a single layer on the prepared baking sheet and roast for 35 to 40 minutes until caramelized.",
    "Roast the chicken in the preheated oven for 45 minutes until the internal temperature reaches 165 degrees F.",

    # --- Soups, Stews, & Sauces ---
    "In a large soup pot, heat olive oil over medium heat and add diced carrots, celery, and onions.",
    "Pour in low-sodium chicken broth and bring the mixture to a gentle simmer.",
    "Pour in vegetable broth, crushed tomatoes, and dried oregano, stirring to combine.",
    "Reduce heat to low, cover with a tight-fitting lid, and simmer gently for 45 minutes.",
    "Cover the Dutch oven with a heavy lid and braise in the oven for 2 hours until the meat is fork-tender.",
    "Stir in heavy cream and freshly grated Parmesan cheese until the sauce is velvety and smooth.",
    "Stir in sour cream, Dijon mustard, and chopped dill until thoroughly combined.",
    "Whisk flour and melted butter together to form a smooth roux, cooking for 2 minutes until lightly golden.",
    "Slowly pour in warm whole milk while whisking constantly to prevent lumps from forming.",
    "Simmer the tomato sauce over low heat, stirring occasionally, for 20 minutes until thickened.",
    "Blend the cooked soup with an immersion blender until completely smooth and velvety.",
    "Purée the soup in batches using a countertop blender until completely smooth, taking care with hot liquids.",

    # --- Seasoning, Garnishing, & Finishing ---
    "Season both sides of the salmon fillets generously with kosher salt, black pepper, and paprika.",
    "Season the soup with kosher salt, freshly cracked black pepper, and a pinch of cayenne pepper.",
    "Season to taste with salt, black pepper, and a squeeze of fresh lemon juice.",
    "Garnish the finished dish with freshly chopped flat-leaf parsley, toasted pine nuts, and lemon wedges.",
    "Garnish with freshly sliced scallions, toasted sesame seeds, and a drizzle of chili oil.",
    "Drizzle with extra virgin olive oil and a splash of aged balsamic vinegar just before serving.",
    "Drizzle the warm glaze over the cooled cake and let set before slicing.",
    "Sprinkle with red pepper flakes and grated pecorino Romano cheese for added heat and flavor.",
    "Sprinkle fresh chopped cilantro and crumbled cotija cheese over the warm tacos.",
    "Let the roasted meat rest on a cutting board for 10 minutes before slicing against the grain.",
    "Let the roast rest for 15 minutes under a loose tent of aluminum foil.",

    # --- Marinades, Dressings, & Salads ---
    "In a small bowl, whisk together soy sauce, toasted sesame oil, minced fresh ginger, and honey.",
    "In a medium bowl, whisk together balsamic vinegar, honey, Dijon mustard, minced garlic, and olive oil.",
    "Marinate the sliced flank steak in the soy-ginger marinade in the refrigerator for at least 1 hour.",
    "Marinate the chicken thighs in Greek yogurt, lemon juice, garlic, and oregano for 2 hours.",
    "In a glass jar, combine olive oil, Dijon mustard, red wine vinegar, minced shallot, and salt, then shake vigorously.",
    "Toss crisp mixed salad greens, sliced cucumbers, and cherry tomatoes with the vinaigrette dressing.",
    "Dice the ripe avocados and gently fold them into the tomato salsa with fresh lime juice and chopped cilantro.",
    "Toss shredded cabbage and carrots with mayonnaise, apple cider vinegar, and sugar for a crisp coleslaw.",

    # --- Breakfast & Light Meals ---
    "Whisk eggs, milk, salt, and black pepper together in a shallow dish for French toast.",
    "Whisk large eggs, a splash of milk, salt, and pepper in a bowl until light and uniform.",
    "Cook scrambled eggs gently in butter over low heat, stirring slowly until soft curds form.",
    "Flip the pancakes carefully when bubbles form on the surface and the edges look set.",
    "Toast the bread slices until golden brown and spread with ripe mashed avocado and sea salt.",
    "Top the toasted bagel with cream cheese, thinly sliced smoked salmon, red onion, and capers."
]


def ensure_data_dir() -> Path:
    """Ensure data directory exists and return its path."""
    DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_DATA_DIR


def write_builtin_corpus_file(target_file: Optional[Path] = None) -> Path:
    """Write the built-in corpus into a local text file if not already present."""
    ensure_data_dir()
    path = target_file or DEFAULT_RECIPES_FILE
    if not path.exists():
        with open(path, "w", encoding="utf-8") as f:
            for step in BUILTIN_RECIPE_STEPS:
                f.write(step.strip() + "\n")
    return path


def load_recipe_corpus(source_path: Optional[str | Path] = None) -> List[str]:
    """Load recipe instruction steps from a file or fallback to the built-in corpus.

    Supports:
    - Plain text files (one recipe instruction per line, ignoring blank lines)
    - JSON files (list of strings or list of dicts with 'instruction'/'step' key)
    - None / default: loads built-in curated recipe steps (and creates recipes.txt)
    """
    if source_path is None:
        target = DEFAULT_RECIPES_FILE
        if not target.exists():
            write_builtin_corpus_file(target)
        source_path = target

    path = Path(source_path)
    if not path.exists():
        # Fallback to built-in if specified file does not exist
        return list(BUILTIN_RECIPE_STEPS)

    recipe_steps: List[str] = []
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, str):
                        step = item.strip()
                        if step:
                            recipe_steps.append(step)
                    elif isinstance(item, dict):
                        step = item.get("step") or item.get("instruction") or item.get("text")
                        if step and isinstance(step, str):
                            recipe_steps.append(step.strip())
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                clean_line = line.strip()
                if clean_line and not clean_line.startswith("#"):
                    recipe_steps.append(clean_line)

    return recipe_steps if recipe_steps else list(BUILTIN_RECIPE_STEPS)


def append_recipe_step(step: str, target_file: Optional[Path] = None) -> None:
    """Append a new recipe instruction step to the corpus file."""
    path = target_file or DEFAULT_RECIPES_FILE
    ensure_data_dir()
    clean_step = step.strip()
    if not clean_step:
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write(clean_step + "\n")


def save_model_artifact(model_data: dict, filepath: str | Path) -> None:
    """Save serialized model counts and configuration to a JSON file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(model_data, f, indent=2)


def load_model_artifact(filepath: str | Path) -> dict:
    """Load serialized model counts and configuration from a JSON file."""
    path = Path(filepath)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
