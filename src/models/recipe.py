from enum import Enum
from pydantic import BaseModel, ConfigDict, Field


class RecipeCategory(str, Enum):
    APPETIZER = "appetizer"
    MAIN_DISH = "main_dish"
    SECOND_PLATE = "second_plate"
    DESSERT = "dessert"


class DietaryTag(str, Enum):
    VEGAN = "vegan"
    VEGETARIAN = "vegetarian"
    GLUTEN_FREE = "gluten_free"
    DAIRY_FREE = "dairy_free"
    NUT_FREE = "nut_free"
    LOW_CARB = "low_carb"
    TRADITIONAL = "traditional"


class RecipeDifficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Recipe(BaseModel):
    id: str
    name: str
    description: str
    category: RecipeCategory

    ingredients: list[str] = Field(default_factory=list)
    instructions: list[str] = Field(default_factory=list)

    servings: int
    prep_time_minutes: int | None = None
    cook_time_minutes: int | None = None

    dietary_tags: list[DietaryTag] = Field(default_factory=list)
    allergens: list[str] = Field(default_factory=list)

    difficulty: RecipeDifficulty
    is_christmas_traditional: bool | None = None

    def is_suitable_for(self, dietary_requirements: list[DietaryTag]) -> bool:
        return all(tag in self.dietary_tags for tag in dietary_requirements)

    def contains_allergens(self, allergens: list[str]) -> bool:
        allergens_lower = {a.lower() for a in self.allergens}
        return any(allergen.lower() in allergens_lower for allergen in allergens)

    def total_time_minutes(self) -> int:
        return (self.prep_time_minutes or 0) + (self.cook_time_minutes or 0)

    def to_search_text(self) -> str:
        dietary_text = (
            ", ".join(tag.value for tag in self.dietary_tags)
            if self.dietary_tags
            else "no special diet"
        )
        return (
            f"{self.name}. {self.description}. "
            f"Category: {self.category.value}. "
            f"Diet: {dietary_text}. "
            f"Ingredients: {', '.join(self.ingredients)}"
        )

    model_config = ConfigDict(use_enum_values=True)
