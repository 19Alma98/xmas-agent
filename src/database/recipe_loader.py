"""
Recipe loader for populating the vector database.
"""

import json
from pathlib import Path

from ..models.recipe import Recipe
from .vector_store import RecipeVectorStore


class RecipeLoader:
    """
    Utility class to load recipes from various sources into the vector store.
    """

    def __init__(self, vector_store: RecipeVectorStore | None = None):
        """
        Initialize the recipe loader.

        Args:
            vector_store: Vector store instance. Creates new one if None.
        """
        self.vector_store = vector_store or RecipeVectorStore()

    def load_from_json_file(self, file_path: str) -> int:
        """
        Load recipes from a JSON file.

        Args:
            file_path: Path to the JSON file

        Returns:
            Number of recipes loaded
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Recipe file not found: {file_path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        recipes = []
        recipe_list = data if isinstance(data, list) else data.get("recipes", [])

        for recipe_data in recipe_list:
            try:
                recipe = Recipe.model_validate(recipe_data)
                recipes.append(recipe)
            except Exception as e:
                print(f"Error parsing recipe: {e}")
                continue

        if recipes:
            self.vector_store.add_recipes(recipes)

        return len(recipes)

    def load_from_dict_list(self, recipes_data: list[dict]) -> int:
        """
        Load recipes from a list of dictionaries.

        Args:
            recipes_data: List of recipe dictionaries

        Returns:
            Number of recipes loaded
        """
        recipes = []
        for recipe_data in recipes_data:
            try:
                recipe = Recipe.model_validate(recipe_data)
                recipes.append(recipe)
            except Exception as e:
                print(f"Error parsing recipe: {e}")
                continue

        if recipes:
            self.vector_store.add_recipes(recipes)

        return len(recipes)

    def load_sample_recipes(self) -> int:
        """
        Load sample Christmas recipes for testing.

        Returns:
            Number of recipes loaded
        """
        sample_recipes = [
            # Appetizers
            {
                "id": "app_001",
                "name": "Bruschetta al Pomodoro",
                "description": "Classic Italian bruschetta with fresh tomatoes, basil, and garlic on toasted bread",
                "category": "appetizer",
                "ingredients": [
                    "Bread",
                    "Tomatoes",
                    "Fresh basil",
                    "Garlic",
                    "Olive oil",
                    "Salt",
                ],
                "instructions": [
                    "Toast bread slices until golden",
                    "Dice tomatoes and mix with chopped basil and garlic",
                    "Drizzle with olive oil and season with salt",
                    "Top bread with tomato mixture",
                ],
                "servings": 8,
                "prep_time_minutes": 15,
                "cook_time_minutes": 5,
                "dietary_tags": ["vegan", "vegetarian"],
                "allergens": ["gluten"],
                "difficulty": "easy",
                "is_christmas_traditional": False,
            },
            {
                "id": "app_002",
                "name": "Smoked Salmon Canapés",
                "description": "Elegant canapés with smoked salmon, cream cheese, and dill",
                "category": "appetizer",
                "ingredients": [
                    "Smoked salmon",
                    "Cream cheese",
                    "Dill",
                    "Capers",
                    "Crackers",
                    "Lemon",
                ],
                "instructions": [
                    "Spread cream cheese on crackers",
                    "Top with smoked salmon",
                    "Garnish with dill, capers, and lemon zest",
                ],
                "servings": 12,
                "prep_time_minutes": 20,
                "cook_time_minutes": 0,
                "dietary_tags": [],
                "allergens": ["fish", "dairy", "gluten"],
                "difficulty": "easy",
                "is_christmas_traditional": True,
            },
            {
                "id": "app_003",
                "name": "Stuffed Mushrooms",
                "description": "Mushroom caps stuffed with herbs, breadcrumbs, and parmesan",
                "category": "appetizer",
                "ingredients": [
                    "Mushrooms",
                    "Breadcrumbs",
                    "Parmesan",
                    "Garlic",
                    "Parsley",
                    "Butter",
                ],
                "instructions": [
                    "Remove mushroom stems and chop finely",
                    "Mix stems with breadcrumbs, parmesan, garlic, and parsley",
                    "Fill mushroom caps with mixture",
                    "Bake at 180°C for 15 minutes",
                ],
                "servings": 8,
                "prep_time_minutes": 20,
                "cook_time_minutes": 15,
                "dietary_tags": ["vegetarian"],
                "allergens": ["gluten", "dairy"],
                "difficulty": "easy",
                "is_christmas_traditional": False,
            },
            {
                "id": "app_004",
                "name": "Vegan Hummus Platter",
                "description": "Creamy homemade hummus served with fresh vegetables and pita bread",
                "category": "appetizer",
                "ingredients": [
                    "Chickpeas",
                    "Tahini",
                    "Lemon",
                    "Garlic",
                    "Olive oil",
                    "Vegetables",
                    "Pita bread",
                ],
                "instructions": [
                    "Blend chickpeas, tahini, lemon juice, and garlic",
                    "Drizzle with olive oil",
                    "Serve with cut vegetables and toasted pita",
                ],
                "servings": 10,
                "prep_time_minutes": 15,
                "cook_time_minutes": 0,
                "dietary_tags": ["vegan", "vegetarian", "dairy_free"],
                "allergens": ["sesame", "gluten"],
                "difficulty": "easy",
                "is_christmas_traditional": False,
            },
            # Main Dishes
            {
                "id": "main_001",
                "name": "Traditional Christmas Lasagna",
                "description": "Rich layered pasta with ragù, béchamel sauce, and cheese",
                "category": "main_dish",
                "ingredients": [
                    "Lasagna sheets",
                    "Ground beef",
                    "Tomato sauce",
                    "Béchamel",
                    "Parmesan",
                    "Mozzarella",
                ],
                "instructions": [
                    "Prepare ragù with ground beef and tomato sauce",
                    "Make béchamel sauce",
                    "Layer lasagna sheets, ragù, béchamel, and cheese",
                    "Bake at 180°C for 45 minutes",
                ],
                "servings": 8,
                "prep_time_minutes": 45,
                "cook_time_minutes": 45,
                "dietary_tags": [],
                "allergens": ["gluten", "dairy", "eggs"],
                "difficulty": "medium",
                "is_christmas_traditional": True,
            },
            {
                "id": "main_002",
                "name": "Tortellini in Brodo",
                "description": "Traditional Italian tortellini served in rich homemade broth",
                "category": "main_dish",
                "ingredients": ["Tortellini", "Beef broth", "Parmesan", "Parsley"],
                "instructions": [
                    "Prepare rich beef broth",
                    "Cook tortellini in the broth",
                    "Serve with grated parmesan and parsley",
                ],
                "servings": 6,
                "prep_time_minutes": 30,
                "cook_time_minutes": 20,
                "dietary_tags": [],
                "allergens": ["gluten", "eggs", "dairy"],
                "difficulty": "medium",
                "is_christmas_traditional": True,
            },
            {
                "id": "main_003",
                "name": "Vegan Mushroom Risotto",
                "description": "Creamy risotto with mixed mushrooms and white wine",
                "category": "main_dish",
                "ingredients": [
                    "Arborio rice",
                    "Mixed mushrooms",
                    "Vegetable broth",
                    "White wine",
                    "Onion",
                    "Olive oil",
                ],
                "instructions": [
                    "Sauté onion and mushrooms",
                    "Add rice and toast briefly",
                    "Gradually add broth and wine while stirring",
                    "Cook until creamy",
                ],
                "servings": 6,
                "prep_time_minutes": 15,
                "cook_time_minutes": 25,
                "dietary_tags": ["vegan", "vegetarian", "dairy_free", "gluten_free"],
                "allergens": [],
                "difficulty": "medium",
                "is_christmas_traditional": False,
            },
            {
                "id": "main_004",
                "name": "Seafood Linguine",
                "description": "Pasta with mixed seafood in a light white wine and garlic sauce",
                "category": "main_dish",
                "ingredients": [
                    "Linguine",
                    "Mixed seafood",
                    "Garlic",
                    "White wine",
                    "Cherry tomatoes",
                    "Parsley",
                ],
                "instructions": [
                    "Cook pasta al dente",
                    "Sauté garlic and seafood in olive oil",
                    "Add wine and tomatoes",
                    "Toss with pasta and parsley",
                ],
                "servings": 6,
                "prep_time_minutes": 20,
                "cook_time_minutes": 15,
                "dietary_tags": ["dairy_free"],
                "allergens": ["gluten", "shellfish", "fish"],
                "difficulty": "medium",
                "is_christmas_traditional": True,
            },
            # Second Plates
            {
                "id": "second_001",
                "name": "Roasted Turkey with Herbs",
                "description": "Classic roasted turkey with rosemary, thyme, and garlic",
                "category": "second_plate",
                "ingredients": [
                    "Turkey",
                    "Rosemary",
                    "Thyme",
                    "Garlic",
                    "Butter",
                    "Lemon",
                ],
                "instructions": [
                    "Season turkey with herbs and garlic",
                    "Roast at 160°C for 3-4 hours",
                    "Baste regularly with pan juices",
                    "Rest before carving",
                ],
                "servings": 10,
                "prep_time_minutes": 30,
                "cook_time_minutes": 240,
                "dietary_tags": ["gluten_free"],
                "allergens": ["dairy"],
                "difficulty": "hard",
                "is_christmas_traditional": True,
            },
            {
                "id": "second_002",
                "name": "Beef Tenderloin",
                "description": "Perfectly roasted beef tenderloin with red wine reduction",
                "category": "second_plate",
                "ingredients": [
                    "Beef tenderloin",
                    "Red wine",
                    "Shallots",
                    "Butter",
                    "Thyme",
                    "Garlic",
                ],
                "instructions": [
                    "Sear tenderloin on all sides",
                    "Roast at 200°C to desired doneness",
                    "Make red wine reduction sauce",
                    "Slice and serve with sauce",
                ],
                "servings": 8,
                "prep_time_minutes": 20,
                "cook_time_minutes": 35,
                "dietary_tags": ["gluten_free"],
                "allergens": ["dairy"],
                "difficulty": "medium",
                "is_christmas_traditional": True,
            },
            {
                "id": "second_003",
                "name": "Baked Sea Bass",
                "description": "Whole sea bass baked with lemon, olives, and cherry tomatoes",
                "category": "second_plate",
                "ingredients": [
                    "Sea bass",
                    "Lemon",
                    "Olives",
                    "Cherry tomatoes",
                    "Capers",
                    "White wine",
                ],
                "instructions": [
                    "Score fish and season",
                    "Add lemon, olives, tomatoes, and capers",
                    "Bake at 180°C for 25 minutes",
                    "Drizzle with cooking juices",
                ],
                "servings": 4,
                "prep_time_minutes": 15,
                "cook_time_minutes": 25,
                "dietary_tags": ["gluten_free", "dairy_free"],
                "allergens": ["fish"],
                "difficulty": "medium",
                "is_christmas_traditional": True,
            },
            {
                "id": "second_004",
                "name": "Vegan Wellington",
                "description": "Mushroom and lentil wellington wrapped in flaky pastry",
                "category": "second_plate",
                "ingredients": [
                    "Puff pastry",
                    "Mushrooms",
                    "Lentils",
                    "Spinach",
                    "Onion",
                    "Garlic",
                    "Walnuts",
                ],
                "instructions": [
                    "Sauté mushrooms, onion, and garlic",
                    "Mix with lentils, spinach, and walnuts",
                    "Wrap in puff pastry",
                    "Bake until golden",
                ],
                "servings": 6,
                "prep_time_minutes": 40,
                "cook_time_minutes": 35,
                "dietary_tags": ["vegan", "vegetarian", "dairy_free"],
                "allergens": ["gluten", "nuts"],
                "difficulty": "hard",
                "is_christmas_traditional": False,
            },
            {
                "id": "second_005",
                "name": "Stuffed Bell Peppers",
                "description": "Colorful bell peppers stuffed with rice, vegetables, and herbs",
                "category": "second_plate",
                "ingredients": [
                    "Bell peppers",
                    "Rice",
                    "Tomatoes",
                    "Onion",
                    "Herbs",
                    "Olive oil",
                ],
                "instructions": [
                    "Cook rice with herbs and vegetables",
                    "Hollow out bell peppers",
                    "Fill with rice mixture",
                    "Bake at 180°C for 30 minutes",
                ],
                "servings": 6,
                "prep_time_minutes": 25,
                "cook_time_minutes": 30,
                "dietary_tags": [
                    "vegan",
                    "vegetarian",
                    "gluten_free",
                    "dairy_free",
                    "nut_free",
                ],
                "allergens": [],
                "difficulty": "easy",
                "is_christmas_traditional": False,
            },
            # Desserts
            {
                "id": "dessert_001",
                "name": "Panettone",
                "description": "Traditional Italian Christmas sweet bread with candied fruits and raisins",
                "category": "dessert",
                "ingredients": [
                    "Flour",
                    "Eggs",
                    "Butter",
                    "Sugar",
                    "Candied fruits",
                    "Raisins",
                    "Yeast",
                ],
                "instructions": [
                    "Prepare enriched dough with eggs and butter",
                    "Let rise twice",
                    "Add candied fruits and raisins",
                    "Bake in panettone mold",
                ],
                "servings": 12,
                "prep_time_minutes": 60,
                "cook_time_minutes": 50,
                "dietary_tags": ["vegetarian"],
                "allergens": ["gluten", "eggs", "dairy"],
                "difficulty": "hard",
                "is_christmas_traditional": True,
            },
            {
                "id": "dessert_002",
                "name": "Tiramisu",
                "description": "Classic Italian dessert with espresso-soaked ladyfingers and mascarpone cream",
                "category": "dessert",
                "ingredients": [
                    "Ladyfingers",
                    "Mascarpone",
                    "Eggs",
                    "Espresso",
                    "Cocoa",
                    "Sugar",
                ],
                "instructions": [
                    "Make mascarpone cream with egg yolks",
                    "Dip ladyfingers in espresso",
                    "Layer cream and ladyfingers",
                    "Dust with cocoa powder",
                ],
                "servings": 8,
                "prep_time_minutes": 30,
                "cook_time_minutes": 0,
                "dietary_tags": ["vegetarian"],
                "allergens": ["gluten", "eggs", "dairy"],
                "difficulty": "medium",
                "is_christmas_traditional": True,
            },
            {
                "id": "dessert_003",
                "name": "Vegan Chocolate Mousse",
                "description": "Rich and silky chocolate mousse made with aquafaba",
                "category": "dessert",
                "ingredients": ["Dark chocolate", "Aquafaba", "Sugar", "Vanilla"],
                "instructions": [
                    "Melt dark chocolate",
                    "Whip aquafaba to stiff peaks",
                    "Fold chocolate into aquafaba",
                    "Chill for 4 hours",
                ],
                "servings": 6,
                "prep_time_minutes": 20,
                "cook_time_minutes": 0,
                "dietary_tags": ["vegan", "vegetarian", "dairy_free", "gluten_free"],
                "allergens": [],
                "difficulty": "medium",
                "is_christmas_traditional": False,
            },
            {
                "id": "dessert_004",
                "name": "Pandoro",
                "description": "Star-shaped Italian Christmas cake dusted with powdered sugar",
                "category": "dessert",
                "ingredients": [
                    "Flour",
                    "Eggs",
                    "Butter",
                    "Sugar",
                    "Vanilla",
                    "Yeast",
                    "Powdered sugar",
                ],
                "instructions": [
                    "Prepare rich dough with lots of butter",
                    "Let rise multiple times",
                    "Bake in star-shaped mold",
                    "Dust generously with powdered sugar",
                ],
                "servings": 10,
                "prep_time_minutes": 90,
                "cook_time_minutes": 45,
                "dietary_tags": ["vegetarian"],
                "allergens": ["gluten", "eggs", "dairy"],
                "difficulty": "hard",
                "is_christmas_traditional": True,
            },
            {
                "id": "dessert_005",
                "name": "Fruit Tart",
                "description": "Buttery tart shell filled with pastry cream and fresh seasonal fruits",
                "category": "dessert",
                "ingredients": [
                    "Flour",
                    "Butter",
                    "Eggs",
                    "Sugar",
                    "Milk",
                    "Vanilla",
                    "Fresh fruits",
                ],
                "instructions": [
                    "Make shortcrust pastry and blind bake",
                    "Prepare vanilla pastry cream",
                    "Fill tart shell with cream",
                    "Arrange fresh fruits on top",
                ],
                "servings": 8,
                "prep_time_minutes": 45,
                "cook_time_minutes": 25,
                "dietary_tags": ["vegetarian"],
                "allergens": ["gluten", "eggs", "dairy"],
                "difficulty": "medium",
                "is_christmas_traditional": False,
            },
            {
                "id": "dessert_006",
                "name": "Gluten-Free Almond Cake",
                "description": "Moist almond cake made with almond flour and orange zest",
                "category": "dessert",
                "ingredients": [
                    "Almond flour",
                    "Eggs",
                    "Sugar",
                    "Orange zest",
                    "Butter",
                    "Baking powder",
                ],
                "instructions": [
                    "Beat eggs and sugar until fluffy",
                    "Fold in almond flour and orange zest",
                    "Pour into cake pan",
                    "Bake at 170°C for 35 minutes",
                ],
                "servings": 8,
                "prep_time_minutes": 20,
                "cook_time_minutes": 35,
                "dietary_tags": ["vegetarian", "gluten_free"],
                "allergens": ["eggs", "dairy", "nuts"],
                "difficulty": "easy",
                "is_christmas_traditional": False,
            },
        ]

        return self.load_from_dict_list(sample_recipes)
