from datapizza.tools import tool  # type: ignore

from ..database.vector_store import RecipeVectorStore
from ..models.recipe import Recipe, RecipeCategory


# Global vector store instance (initialized lazily)
_vector_store: RecipeVectorStore | None = None


def get_vector_store() -> RecipeVectorStore:
    """Get or create the vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = RecipeVectorStore()
    return _vector_store


def set_vector_store(store: RecipeVectorStore) -> None:
    """Set a custom vector store instance (useful for testing)."""
    global _vector_store
    _vector_store = store


def _format_recipes_for_agent(recipes: list[Recipe]) -> str:
    """Format recipes list as a string for agent consumption."""
    if not recipes:
        return "No recipes found matching the criteria."

    result = []
    for i, recipe in enumerate(recipes, 1):
        dietary = ", ".join(recipe.dietary_tags) if recipe.dietary_tags else "none"
        allergens = ", ".join(recipe.allergens) if recipe.allergens else "none"
        result.append(
            f"{i}. **{recipe.name}** (ID: {recipe.id})\n"
            f"   Description: {recipe.description}\n"
            f"   Dietary tags: {dietary}\n"
            f"   Allergens: {allergens}\n"
            f"   Prep time: {recipe.prep_time_minutes} min | Cook time: {recipe.cook_time_minutes} min\n"
            f"   Servings: {recipe.servings} | Difficulty: {recipe.difficulty}\n"
            f"   Traditional Christmas: {'Yes' if recipe.is_christmas_traditional else 'No'}"
        )

    return "\n\n".join(result)


@tool
def search_appetizers(
    query: str,
    is_vegan: bool = False,
    is_vegetarian: bool = False,
    is_gluten_free: bool = False,
    is_dairy_free: bool = False,
    is_nut_free: bool = False,
    max_prep_time: int | None = None,
    prefer_traditional: bool = False,
    n_results: int = 3,
) -> str:
    """
    Search for appetizer recipes in the database using semantic search with filters.

    Args:
        query: Natural language description of what you're looking for (e.g., "light vegetable appetizers")
        is_vegan: Filter for vegan recipes only
        is_vegetarian: Filter for vegetarian recipes only
        is_gluten_free: Filter for gluten-free recipes only
        is_dairy_free: Filter for dairy-free recipes only
        is_nut_free: Filter for nut-free recipes only
        max_prep_time: Maximum preparation time in minutes
        prefer_traditional: Prefer traditional Christmas recipes
        n_results: Number of results to return (default: 3)

    Returns:
        Formatted string with matching appetizer recipes
    """
    store = get_vector_store()
    recipes = store.search_recipes(
        query=query,
        category=RecipeCategory.APPETIZER,
        n_results=n_results,
        is_vegan=is_vegan if is_vegan else None,
        is_vegetarian=is_vegetarian if is_vegetarian else None,
        is_gluten_free=is_gluten_free if is_gluten_free else None,
        is_dairy_free=is_dairy_free if is_dairy_free else None,
        is_nut_free=is_nut_free if is_nut_free else None,
        max_prep_time=max_prep_time,
        prefer_traditional=prefer_traditional,
    )
    return _format_recipes_for_agent(recipes)


@tool
def search_main_dishes(
    query: str,
    is_vegan: bool = False,
    is_vegetarian: bool = False,
    is_gluten_free: bool = False,
    is_dairy_free: bool = False,
    is_nut_free: bool = False,
    max_prep_time: int | None = None,
    prefer_traditional: bool = False,
    n_results: int = 3,
) -> str:
    """
    Search for main dish recipes in the database using semantic search with filters.

    Args:
        query: Natural language description of what you're looking for (e.g., "pasta dishes for family dinner")
        is_vegan: Filter for vegan recipes only
        is_vegetarian: Filter for vegetarian recipes only
        is_gluten_free: Filter for gluten-free recipes only
        is_dairy_free: Filter for dairy-free recipes only
        is_nut_free: Filter for nut-free recipes only
        max_prep_time: Maximum preparation time in minutes
        prefer_traditional: Prefer traditional Christmas recipes
        n_results: Number of results to return (default: 3)

    Returns:
        Formatted string with matching main dish recipes
    """
    store = get_vector_store()
    recipes = store.search_recipes(
        query=query,
        category=RecipeCategory.MAIN_DISH,
        n_results=n_results,
        is_vegan=is_vegan if is_vegan else None,
        is_vegetarian=is_vegetarian if is_vegetarian else None,
        is_gluten_free=is_gluten_free if is_gluten_free else None,
        is_dairy_free=is_dairy_free if is_dairy_free else None,
        is_nut_free=is_nut_free if is_nut_free else None,
        max_prep_time=max_prep_time,
        prefer_traditional=prefer_traditional,
    )
    return _format_recipes_for_agent(recipes)


@tool
def search_second_plates(
    query: str,
    is_vegan: bool = False,
    is_vegetarian: bool = False,
    is_gluten_free: bool = False,
    is_dairy_free: bool = False,
    is_nut_free: bool = False,
    max_prep_time: int | None = None,
    prefer_traditional: bool = False,
    n_results: int = 3,
) -> str:
    """
    Search for second plate (secondo piatto) recipes in the database using semantic search with filters.
    Second plates are typically meat or fish dishes served after the first course.

    Args:
        query: Natural language description of what you're looking for (e.g., "roasted meat for special occasion")
        is_vegan: Filter for vegan recipes only
        is_vegetarian: Filter for vegetarian recipes only
        is_gluten_free: Filter for gluten-free recipes only
        is_dairy_free: Filter for dairy-free recipes only
        is_nut_free: Filter for nut-free recipes only
        max_prep_time: Maximum preparation time in minutes
        prefer_traditional: Prefer traditional Christmas recipes
        n_results: Number of results to return (default: 3)

    Returns:
        Formatted string with matching second plate recipes
    """
    store = get_vector_store()
    recipes = store.search_recipes(
        query=query,
        category=RecipeCategory.SECOND_PLATE,
        n_results=n_results,
        is_vegan=is_vegan if is_vegan else None,
        is_vegetarian=is_vegetarian if is_vegetarian else None,
        is_gluten_free=is_gluten_free if is_gluten_free else None,
        is_dairy_free=is_dairy_free if is_dairy_free else None,
        is_nut_free=is_nut_free if is_nut_free else None,
        max_prep_time=max_prep_time,
        prefer_traditional=prefer_traditional,
    )
    return _format_recipes_for_agent(recipes)


@tool
def search_desserts(
    query: str,
    is_vegan: bool = False,
    is_vegetarian: bool = False,
    is_gluten_free: bool = False,
    is_dairy_free: bool = False,
    is_nut_free: bool = False,
    max_prep_time: int | None = None,
    prefer_traditional: bool = False,
    n_results: int = 3,
) -> str:
    """
    Search for dessert recipes in the database using semantic search with filters.

    Args:
        query: Natural language description of what you're looking for (e.g., "chocolate dessert for Christmas")
        is_vegan: Filter for vegan recipes only
        is_vegetarian: Filter for vegetarian recipes only
        is_gluten_free: Filter for gluten-free recipes only
        is_dairy_free: Filter for dairy-free recipes only
        is_nut_free: Filter for nut-free recipes only
        max_prep_time: Maximum preparation time in minutes
        prefer_traditional: Prefer traditional Christmas recipes
        n_results: Number of results to return (default: 3)

    Returns:
        Formatted string with matching dessert recipes
    """
    store = get_vector_store()
    recipes = store.search_recipes(
        query=query,
        category=RecipeCategory.DESSERT,
        n_results=n_results,
        is_vegan=is_vegan if is_vegan else None,
        is_vegetarian=is_vegetarian if is_vegetarian else None,
        is_gluten_free=is_gluten_free if is_gluten_free else None,
        is_dairy_free=is_dairy_free if is_dairy_free else None,
        is_nut_free=is_nut_free if is_nut_free else None,
        max_prep_time=max_prep_time,
        prefer_traditional=prefer_traditional,
    )
    return _format_recipes_for_agent(recipes)


@tool
def get_recipe_details(recipe_id: str) -> str:
    """
    Get detailed information about a specific recipe by its ID.

    Args:
        recipe_id: The unique identifier of the recipe (e.g., "app_001", "main_002")

    Returns:
        Detailed recipe information including all ingredients and instructions
    """
    store = get_vector_store()
    recipe = store.get_recipe_by_id(recipe_id)

    if not recipe:
        return f"Recipe with ID '{recipe_id}' not found."

    dietary = ", ".join(recipe.dietary_tags) if recipe.dietary_tags else "none"
    allergens = ", ".join(recipe.allergens) if recipe.allergens else "none"

    ingredients_list = "\n".join(f"   • {ing}" for ing in recipe.ingredients)
    instructions_list = "\n".join(
        f"   {i}. {step}" for i, step in enumerate(recipe.instructions, 1)
    )

    return f"""
**{recipe.name}** (ID: {recipe.id})
Category: {recipe.category}
Description: {recipe.description}

📊 Details:
   • Servings: {recipe.servings}
   • Prep time: {recipe.prep_time_minutes} minutes
   • Cook time: {recipe.cook_time_minutes} minutes
   • Total time: {recipe.total_time_minutes()} minutes
   • Difficulty: {recipe.difficulty}
   • Traditional Christmas: {"Yes" if recipe.is_christmas_traditional else "No"}

🥗 Dietary Tags: {dietary}
⚠️ Allergens: {allergens}

🛒 Ingredients:
{ingredients_list}

👨‍🍳 Instructions:
{instructions_list}
"""
