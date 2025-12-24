import json
from pathlib import Path
from uuid import uuid4

from datapizza.modules.parsers.docling import DoclingParser  # type: ignore

from ..agents.document_parsing_agent import DocumentParsingAgent
from ..models.recipe import DietaryTag, Recipe, RecipeCategory, RecipeDifficulty
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
            {
                "id": "it_app_001",
                "name": "Antipasto Misto Italiano",
                "description": "Classic Italian mixed appetizer platter with prosciutto, salami, cheese, olives, and roasted vegetables",
                "category": RecipeCategory.APPETIZER.value,
                "ingredients": [
                    "Prosciutto di Parma",
                    "Salame",
                    "Pecorino cheese",
                    "Mozzarella di bufala",
                    "Olive verdi e nere",
                    "Peperoni arrostiti",
                    "Carciofi sott'olio",
                    "Pane tostato",
                ],
                "instructions": [
                    "Disponi su un piatto grande il prosciutto e il salame tagliati a fette",
                    "Aggiungi i formaggi tagliati a cubetti",
                    "Completa con olive, peperoni e carciofi",
                    "Servi con pane tostato",
                ],
                "servings": 8,
                "prep_time_minutes": 20,
                "cook_time_minutes": 0,
                "dietary_tags": [],
                "allergens": ["gluten", "dairy"],
                "difficulty": RecipeDifficulty.EASY.value,
                "is_christmas_traditional": True,
            },
            {
                "id": "it_app_002",
                "name": "Bruschetta al Pomodoro e Basilico",
                "description": "Traditional Italian bruschetta with fresh tomatoes, basil, garlic, and extra virgin olive oil",
                "category": RecipeCategory.APPETIZER.value,
                "ingredients": [
                    "Pane casereccio",
                    "Pomodori maturi",
                    "Aglio",
                    "Basilico fresco",
                    "Olio extravergine d'oliva",
                    "Sale",
                    "Pepe nero",
                ],
                "instructions": [
                    "Taglia il pane a fette spesse e tostalo",
                    "Sfregare le fette con uno spicchio d'aglio",
                    "Taglia i pomodori a cubetti e condisci con olio, sale e basilico",
                    "Disponi il condimento sulle fette di pane",
                ],
                "servings": 6,
                "prep_time_minutes": 15,
                "cook_time_minutes": 5,
                "dietary_tags": [DietaryTag.VEGETARIAN.value, DietaryTag.VEGAN.value],
                "allergens": ["gluten"],
                "difficulty": RecipeDifficulty.EASY.value,
                "is_christmas_traditional": False,
            },
            {
                "id": "it_app_003",
                "name": "Crostini ai Funghi",
                "description": "Toasted bread topped with sautéed mushrooms, garlic, and parsley",
                "category": RecipeCategory.APPETIZER.value,
                "ingredients": [
                    "Pane casereccio",
                    "Funghi porcini",
                    "Aglio",
                    "Prezzemolo",
                    "Olio extravergine d'oliva",
                    "Vino bianco",
                    "Sale",
                    "Pepe",
                ],
                "instructions": [
                    "Taglia il pane a fette sottili e tostalo",
                    "Pulisci e taglia i funghi a fettine",
                    "Soffriggi aglio in olio, aggiungi i funghi",
                    "Sfuma con vino bianco, aggiungi prezzemolo",
                    "Disponi sui crostini",
                ],
                "servings": 8,
                "prep_time_minutes": 20,
                "cook_time_minutes": 15,
                "dietary_tags": [DietaryTag.VEGETARIAN.value],
                "allergens": ["gluten"],
                "difficulty": RecipeDifficulty.EASY.value,
                "is_christmas_traditional": True,
            },
            {
                "id": "it_app_004",
                "name": "Caponata Siciliana",
                "description": "Traditional Sicilian eggplant caponata with tomatoes, olives, capers, and celery",
                "category": RecipeCategory.APPETIZER.value,
                "ingredients": [
                    "Melanzane",
                    "Pomodori",
                    "Cipolla",
                    "Sedano",
                    "Olive nere",
                    "Capperi",
                    "Aceto",
                    "Zucchero",
                    "Olio extravergine d'oliva",
                ],
                "instructions": [
                    "Taglia le melanzane a cubetti e friggi",
                    "Soffriggi cipolla e sedano",
                    "Aggiungi pomodori, olive e capperi",
                    "Condisci con aceto e zucchero",
                    "Lascia raffreddare e servi",
                ],
                "servings": 8,
                "prep_time_minutes": 30,
                "cook_time_minutes": 30,
                "dietary_tags": [DietaryTag.VEGETARIAN.value, DietaryTag.VEGAN.value],
                "allergens": [],
                "difficulty": RecipeDifficulty.MEDIUM.value,
                "is_christmas_traditional": True,
            },
            {
                "id": "it_main_001",
                "name": "Lasagne al Forno",
                "description": "Traditional Italian baked lasagna with ragù, béchamel sauce, and Parmigiano Reggiano",
                "category": RecipeCategory.MAIN_DISH.value,
                "ingredients": [
                    "Pasta per lasagne",
                    "Ragù di carne",
                    "Béchamel",
                    "Parmigiano Reggiano",
                    "Mozzarella",
                    "Burro",
                ],
                "instructions": [
                    "Prepara il ragù con carne macinata e pomodoro",
                    "Prepara la béchamel",
                    "Alterna strati di pasta, ragù, béchamel e formaggio",
                    "Cuoci in forno a 180°C per 45 minuti",
                ],
                "servings": 8,
                "prep_time_minutes": 60,
                "cook_time_minutes": 45,
                "dietary_tags": [],
                "allergens": ["gluten", "dairy", "eggs"],
                "difficulty": RecipeDifficulty.MEDIUM.value,
                "is_christmas_traditional": True,
            },
            {
                "id": "it_main_002",
                "name": "Tortellini in Brodo",
                "description": "Traditional Christmas tortellini served in rich homemade meat broth",
                "category": RecipeCategory.MAIN_DISH.value,
                "ingredients": [
                    "Tortellini",
                    "Brodo di carne",
                    "Parmigiano Reggiano",
                    "Prezzemolo",
                ],
                "instructions": [
                    "Prepara un brodo di carne ricco",
                    "Cuoci i tortellini nel brodo",
                    "Servi con Parmigiano grattugiato e prezzemolo",
                ],
                "servings": 6,
                "prep_time_minutes": 30,
                "cook_time_minutes": 20,
                "dietary_tags": [],
                "allergens": ["gluten", "eggs", "dairy"],
                "difficulty": RecipeDifficulty.MEDIUM.value,
                "is_christmas_traditional": True,
            },
            {
                "id": "it_main_003",
                "name": "Risotto ai Porcini",
                "description": "Creamy risotto with porcini mushrooms, white wine, and Parmigiano",
                "category": RecipeCategory.MAIN_DISH.value,
                "ingredients": [
                    "Riso Carnaroli",
                    "Funghi porcini",
                    "Brodo vegetale",
                    "Vino bianco",
                    "Cipolla",
                    "Parmigiano Reggiano",
                    "Burro",
                    "Olio extravergine d'oliva",
                ],
                "instructions": [
                    "Soffriggi cipolla in olio e burro",
                    "Aggiungi i funghi tagliati",
                    "Tosta il riso, sfuma con vino",
                    "Aggiungi brodo caldo gradualmente mescolando",
                    "Mantecare con burro e Parmigiano",
                ],
                "servings": 6,
                "prep_time_minutes": 15,
                "cook_time_minutes": 25,
                "dietary_tags": [DietaryTag.VEGETARIAN.value],
                "allergens": ["gluten", "dairy"],
                "difficulty": RecipeDifficulty.MEDIUM.value,
                "is_christmas_traditional": True,
            },
            {
                "id": "it_main_004",
                "name": "Cannelloni Ricotta e Spinaci",
                "description": "Baked pasta tubes filled with ricotta, spinach, and herbs, topped with tomato sauce",
                "category": RecipeCategory.MAIN_DISH.value,
                "ingredients": [
                    "Cannelloni",
                    "Ricotta",
                    "Spinaci",
                    "Parmigiano Reggiano",
                    "Salsa di pomodoro",
                    "Noce moscata",
                    "Uova",
                ],
                "instructions": [
                    "Cuoci gli spinaci e strizzali",
                    "Mescola ricotta, spinaci, uova e Parmigiano",
                    "Riempi i cannelloni",
                    "Disponi in teglia con salsa di pomodoro",
                    "Cuoci in forno a 180°C per 30 minuti",
                ],
                "servings": 6,
                "prep_time_minutes": 40,
                "cook_time_minutes": 30,
                "dietary_tags": [DietaryTag.VEGETARIAN.value],
                "allergens": ["gluten", "dairy", "eggs"],
                "difficulty": RecipeDifficulty.MEDIUM.value,
                "is_christmas_traditional": False,
            },
            {
                "id": "it_second_001",
                "name": "Arrosto di Tacchino",
                "description": "Traditional Italian roasted turkey with herbs, garlic, and white wine",
                "category": RecipeCategory.SECOND_PLATE.value,
                "ingredients": [
                    "Tacchino",
                    "Rosmarino",
                    "Salvia",
                    "Aglio",
                    "Vino bianco",
                    "Burro",
                    "Olio extravergine d'oliva",
                ],
                "instructions": [
                    "Marina il tacchino con erbe e aglio",
                    "Rosola in padella con burro e olio",
                    "Aggiungi vino bianco",
                    "Cuoci in forno a 160°C per 3-4 ore",
                    "Bagna con il fondo di cottura",
                ],
                "servings": 10,
                "prep_time_minutes": 30,
                "cook_time_minutes": 240,
                "dietary_tags": [],
                "allergens": [],
                "difficulty": RecipeDifficulty.HARD.value,
                "is_christmas_traditional": True,
            },
            {
                "id": "it_second_002",
                "name": "Filetto di Manzo al Pepe Verde",
                "description": "Beef tenderloin with green peppercorn sauce and red wine reduction",
                "category": RecipeCategory.SECOND_PLATE.value,
                "ingredients": [
                    "Filetto di manzo",
                    "Pepe verde",
                    "Vino rosso",
                    "Burro",
                    "Scalogno",
                    "Brodo di carne",
                    "Panna",
                ],
                "instructions": [
                    "Rosola il filetto su tutti i lati",
                    "Cuoci in forno a 200°C al punto desiderato",
                    "Prepara la salsa con scalogno, vino e brodo",
                    "Aggiungi pepe verde e panna",
                    "Servi il filetto con la salsa",
                ],
                "servings": 8,
                "prep_time_minutes": 20,
                "cook_time_minutes": 35,
                "dietary_tags": [],
                "allergens": ["dairy"],
                "difficulty": RecipeDifficulty.MEDIUM.value,
                "is_christmas_traditional": True,
            },
            {
                "id": "it_second_003",
                "name": "Branzino al Forno",
                "description": "Whole sea bass baked with lemon, herbs, and white wine",
                "category": RecipeCategory.SECOND_PLATE.value,
                "ingredients": [
                    "Branzino",
                    "Limone",
                    "Rosmarino",
                    "Aglio",
                    "Vino bianco",
                    "Olio extravergine d'oliva",
                    "Patate",
                ],
                "instructions": [
                    "Pulisci il pesce e incidi la pelle",
                    "Condisci con limone, erbe e aglio",
                    "Disponi su patate tagliate",
                    "Cuoci in forno a 180°C per 25 minuti",
                    "Bagna con vino bianco durante la cottura",
                ],
                "servings": 4,
                "prep_time_minutes": 15,
                "cook_time_minutes": 25,
                "dietary_tags": [],
                "allergens": ["fish"],
                "difficulty": RecipeDifficulty.MEDIUM.value,
                "is_christmas_traditional": True,
            },
            {
                "id": "it_second_004",
                "name": "Costolette d'Agnello",
                "description": "Herb-crusted lamb chops with rosemary and garlic",
                "category": RecipeCategory.SECOND_PLATE.value,
                "ingredients": [
                    "Costolette d'agnello",
                    "Rosmarino",
                    "Aglio",
                    "Olio extravergine d'oliva",
                    "Sale",
                    "Pepe",
                ],
                "instructions": [
                    "Marina le costolette con olio, rosmarino e aglio",
                    "Rosola in padella calda",
                    "Cuoci al punto desiderato",
                    "Lascia riposare prima di servire",
                ],
                "servings": 6,
                "prep_time_minutes": 15,
                "cook_time_minutes": 20,
                "dietary_tags": [],
                "allergens": [],
                "difficulty": RecipeDifficulty.MEDIUM.value,
                "is_christmas_traditional": True,
            },
            {
                "id": "it_dessert_001",
                "name": "Panettone",
                "description": "Traditional Italian Christmas sweet bread with candied fruits and raisins",
                "category": RecipeCategory.DESSERT.value,
                "ingredients": [
                    "Farina",
                    "Uova",
                    "Burro",
                    "Zucchero",
                    "Frutta candita",
                    "Uvetta",
                    "Lievito",
                    "Vaniglia",
                ],
                "instructions": [
                    "Prepara l'impasto con farina, uova, burro e zucchero",
                    "Lascia lievitare due volte",
                    "Aggiungi frutta candita e uvetta",
                    "Cuoci in stampo per panettone",
                    "Cuoci a 180°C per 50 minuti",
                ],
                "servings": 12,
                "prep_time_minutes": 90,
                "cook_time_minutes": 50,
                "dietary_tags": [DietaryTag.VEGETARIAN.value],
                "allergens": ["gluten", "eggs", "dairy"],
                "difficulty": RecipeDifficulty.HARD.value,
                "is_christmas_traditional": True,
            },
            {
                "id": "it_dessert_002",
                "name": "Pandoro",
                "description": "Star-shaped Italian Christmas cake dusted with powdered sugar",
                "category": RecipeCategory.DESSERT.value,
                "ingredients": [
                    "Farina",
                    "Uova",
                    "Burro",
                    "Zucchero",
                    "Vaniglia",
                    "Lievito",
                    "Zucchero a velo",
                ],
                "instructions": [
                    "Prepara l'impasto ricco con molto burro",
                    "Lascia lievitare più volte",
                    "Cuoci in stampo a stella",
                    "Cuoci a 180°C per 45 minuti",
                    "Spolvera con zucchero a velo",
                ],
                "servings": 10,
                "prep_time_minutes": 90,
                "cook_time_minutes": 45,
                "dietary_tags": [DietaryTag.VEGETARIAN.value],
                "allergens": ["gluten", "eggs", "dairy"],
                "difficulty": RecipeDifficulty.HARD.value,
                "is_christmas_traditional": True,
            },
            {
                "id": "it_dessert_003",
                "name": "Tiramisù",
                "description": "Classic Italian dessert with espresso-soaked ladyfingers and mascarpone cream",
                "category": RecipeCategory.DESSERT.value,
                "ingredients": [
                    "Savoiardi",
                    "Mascarpone",
                    "Uova",
                    "Caffè espresso",
                    "Cacao amaro",
                    "Zucchero",
                ],
                "instructions": [
                    "Prepara la crema con mascarpone e tuorli",
                    "Monta gli albumi e incorpora",
                    "Bagna i savoiardi nel caffè",
                    "Alterna strati di savoiardi e crema",
                    "Spolvera con cacao",
                ],
                "servings": 8,
                "prep_time_minutes": 30,
                "cook_time_minutes": 0,
                "dietary_tags": [DietaryTag.VEGETARIAN.value],
                "allergens": ["gluten", "eggs", "dairy"],
                "difficulty": RecipeDifficulty.MEDIUM.value,
                "is_christmas_traditional": True,
            },
            {
                "id": "it_dessert_004",
                "name": "Struffoli",
                "description": "Traditional Neapolitan Christmas honey balls",
                "category": RecipeCategory.DESSERT.value,
                "ingredients": [
                    "Farina",
                    "Uova",
                    "Burro",
                    "Miele",
                    "Canditi",
                    "Zucchero a velo",
                ],
                "instructions": [
                    "Prepara l'impasto con farina, uova e burro",
                    "Forma piccole palline",
                    "Friggi le palline",
                    "Condisci con miele caldo",
                    "Decora con canditi e zucchero a velo",
                ],
                "servings": 10,
                "prep_time_minutes": 60,
                "cook_time_minutes": 30,
                "dietary_tags": [DietaryTag.VEGETARIAN.value],
                "allergens": ["gluten", "eggs", "dairy"],
                "difficulty": RecipeDifficulty.HARD.value,
                "is_christmas_traditional": True,
            },
        ]

        return self.load_from_dict_list(sample_recipes)

    def load_from_pdf(
        self,
        file_path: str,
        api_key: str | None = None,
        provider: str | None = None,
    ) -> int:
        """
        Load recipes from a PDF file using datapizza AI DoclingParser and DocumentParsingAgent.
        Extracts text from PDF, uses AI agent to extract recipe information,
        and stores as text chunks with metadata in the vector store.

        Args:
            file_path: Path to the PDF file
            api_key: OpenAI API key for the parsing agent (optional)
            provider: LLM provider override ("openai" or "ollama")

        Returns:
            Number of recipes loaded

        Raises:
            FileNotFoundError: If PDF file doesn't exist
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        try:
            parser = DoclingParser()
            document_node = parser.parse(str(path))
            full_text = self._extract_text_from_document_node(document_node)
        except Exception as e:
            raise ValueError(f"Error reading PDF: {e}") from e

        parsing_agent = DocumentParsingAgent(api_key=api_key, provider=provider)
        extracted_recipes = parsing_agent.extract_recipes(full_text)

        chunks = []
        for recipe_data in extracted_recipes:
            recipe_text = self._recipe_dict_to_text(recipe_data)

            metadata = {
                "source": "pdf",
                "source_file": str(path.name),
                "category": recipe_data.get("category", "main_dish"),
                "is_christmas_traditional": recipe_data.get(
                    "is_christmas_traditional", True
                ),
                "difficulty": recipe_data.get("difficulty", "medium"),
                "servings": recipe_data.get("servings", 4),
                "prep_time_minutes": recipe_data.get("prep_time_minutes"),
                "cook_time_minutes": recipe_data.get("cook_time_minutes"),
                "dietary_tags": json.dumps(recipe_data.get("dietary_tags", [])),
                "allergens": json.dumps(recipe_data.get("allergens", [])),
                "recipe_name": recipe_data.get("name", "Unknown Recipe"),
                "recipe_json": json.dumps(recipe_data),
            }

            dietary_tags = recipe_data.get("dietary_tags", [])
            metadata["is_vegan"] = "vegan" in dietary_tags
            metadata["is_vegetarian"] = "vegetarian" in dietary_tags
            metadata["is_gluten_free"] = "gluten_free" in dietary_tags
            metadata["is_dairy_free"] = "dairy_free" in dietary_tags
            metadata["is_nut_free"] = "nut_free" in dietary_tags

            chunks.append(
                {
                    "id": f"pdf_{uuid4().hex[:12]}_{recipe_data.get('name', 'recipe').lower().replace(' ', '_')[:20]}",
                    "text": recipe_text,
                    "metadata": metadata,
                }
            )

        if chunks:
            self.vector_store.add_text_chunks(chunks)

        return len(chunks)

    def _recipe_dict_to_text(self, recipe_data: dict) -> str:
        """
        Convert a recipe dictionary to a searchable text representation.

        Args:
            recipe_data: Dictionary with recipe information

        Returns:
            Formatted text string
        """
        name = recipe_data.get("name", "Unknown Recipe")
        description = recipe_data.get("description", "")
        category = recipe_data.get("category", "main_dish")
        ingredients = recipe_data.get("ingredients", [])
        instructions = recipe_data.get("instructions", [])
        dietary_tags = recipe_data.get("dietary_tags", [])

        dietary_text = ", ".join(dietary_tags) if dietary_tags else "no special diet"
        ingredients_text = ", ".join(ingredients) if ingredients else "See recipe"
        instructions_text = " ".join(instructions) if instructions else "See recipe"

        return (
            f"{name}. {description}. "
            f"Category: {category}. "
            f"Diet: {dietary_text}. "
            f"Ingredients: {ingredients_text}. "
            f"Instructions: {instructions_text}"
        )

    def _extract_text_from_document_node(self, node) -> str:
        """
        Extract text content from a datapizza document node.
        Recursively traverses the document structure to collect all text.

        Args:
            node: Document node from DoclingParser

        Returns:
            Extracted text content
        """
        text_parts = []

        if hasattr(node, "content") and node.content:
            text_parts.append(str(node.content))

        if hasattr(node, "children") and node.children:
            for child in node.children:
                child_text = self._extract_text_from_document_node(child)
                if child_text:
                    text_parts.append(child_text)

        return "\n".join(text_parts)
