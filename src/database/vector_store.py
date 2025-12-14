import json

from datapizza.vectorstores.qdrant import QdrantVectorstore  # type: ignore
from datapizza.core.vectorstore import VectorConfig, Distance  # type: ignore
from datapizza.type import Chunk, DenseEmbedding  # type: ignore
from datapizza.embedders.openai import OpenAIEmbedder  # type: ignore

from ..config.settings import settings
from ..models.recipe import Recipe, RecipeCategory, DietaryTag


class RecipeVectorStore:
    """
    Vector store for recipes.
    Supports filtered similarity search for recipe retrieval.
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        location: str | None = None,
    ):
        """
        Initialize the vector store.

        Args:
            host: Qdrant server host. Uses settings default if None.
            port: Qdrant server port. Uses settings default if None.
            location: In-memory or file-based location. Uses settings default if None.
        """
        self.collection_name = settings.COLLECTION_NAME
        self.embedding_name = "recipe_embedding"

        self.embedder = OpenAIEmbedder(
            api_key=settings.OPENAI_API_KEY,
            model_name=settings.EMBEDDING_MODEL,
        )

        host = host or settings.QDRANT_HOST
        port = port or settings.QDRANT_PORT
        location = location or settings.QDRANT_LOCATION

        if host:
            self.vectorstore = QdrantVectorstore(
                host=host,
                port=port,
            )
        else:
            self.vectorstore = QdrantVectorstore(location=location)

        self._ensure_collection_exists()

    def _ensure_collection_exists(self) -> None:
        """Ensure the recipes collection exists."""
        try:
            self.vectorstore.create_collection(
                collection_name=self.collection_name,
                vector_config=[
                    VectorConfig(
                        name=self.embedding_name,
                        dimensions=settings.EMBEDDING_DIMENSIONS,
                        distance=Distance.COSINE,
                    )
                ],
            )
        except Exception:
            pass

    def _recipe_to_metadata(self, recipe: Recipe) -> dict:
        """Convert recipe to metadata dictionary for storage."""
        return {
            "category": recipe.category
            if isinstance(recipe.category, str)
            else recipe.category.value,
            "is_vegan": "vegan"
            in [
                t.value if isinstance(t, DietaryTag) else t for t in recipe.dietary_tags
            ],
            "is_vegetarian": "vegetarian"
            in [
                t.value if isinstance(t, DietaryTag) else t for t in recipe.dietary_tags
            ],
            "is_gluten_free": "gluten_free"
            in [
                t.value if isinstance(t, DietaryTag) else t for t in recipe.dietary_tags
            ],
            "is_dairy_free": "dairy_free"
            in [
                t.value if isinstance(t, DietaryTag) else t for t in recipe.dietary_tags
            ],
            "is_nut_free": "nut_free"
            in [
                t.value if isinstance(t, DietaryTag) else t for t in recipe.dietary_tags
            ],
            "is_christmas_traditional": recipe.is_christmas_traditional,
            "difficulty": recipe.difficulty,
            "prep_time_minutes": recipe.prep_time_minutes,
            "cook_time_minutes": recipe.cook_time_minutes,
            "servings": recipe.servings,
            "allergens": json.dumps(recipe.allergens),
            "dietary_tags": json.dumps(
                [
                    t.value if isinstance(t, DietaryTag) else t
                    for t in recipe.dietary_tags
                ]
            ),
            "recipe_json": recipe.model_dump_json(),
        }

    def _create_chunk(self, recipe: Recipe, embedding: list[float]) -> Chunk:
        """Create a Chunk object from a recipe."""
        metadata = self._recipe_to_metadata(recipe)
        search_text = recipe.to_search_text()

        return Chunk(
            id=recipe.id,
            text=search_text,
            metadata=metadata,
            embeddings=[DenseEmbedding(name=self.embedding_name, vector=embedding)],
        )

    def add_recipe(self, recipe: Recipe) -> None:
        """
        Add a single recipe to the vector store.

        Args:
            recipe: Recipe to add
        """
        search_text = recipe.to_search_text()
        embedding = self.embedder.embed(search_text)

        chunk = self._create_chunk(recipe, embedding)
        self.vectorstore.add([chunk], collection_name=self.collection_name)

    def add_recipes(self, recipes: list[Recipe]) -> None:
        """
        Add multiple recipes to the vector store.

        Args:
            recipes: List of recipes to add
        """
        if not recipes:
            return

        search_texts = [recipe.to_search_text() for recipe in recipes]

        embeddings = self.embedder.embed(search_texts)

        chunks = []
        for recipe, embedding in zip(recipes, embeddings):
            chunk = self._create_chunk(recipe, embedding)
            chunks.append(chunk)

        self.vectorstore.add(chunks, collection_name=self.collection_name)

    def _build_filter(
        self,
        category: RecipeCategory | None = None,
        is_vegan: bool | None = None,
        is_vegetarian: bool | None = None,
        is_gluten_free: bool | None = None,
        is_dairy_free: bool | None = None,
        is_nut_free: bool | None = None,
        max_prep_time: int | None = None,
        prefer_traditional: bool = False,
    ) -> dict | None:
        """Build filter dictionary for Qdrant search."""
        conditions = []

        if category:
            cat_value = category if isinstance(category, str) else category.value
            conditions.append({"key": "category", "match": {"value": cat_value}})

        if is_vegan is True:
            conditions.append({"key": "is_vegan", "match": {"value": True}})

        if is_vegetarian is True:
            conditions.append({"key": "is_vegetarian", "match": {"value": True}})

        if is_gluten_free is True:
            conditions.append({"key": "is_gluten_free", "match": {"value": True}})

        if is_dairy_free is True:
            conditions.append({"key": "is_dairy_free", "match": {"value": True}})

        if is_nut_free is True:
            conditions.append({"key": "is_nut_free", "match": {"value": True}})

        if max_prep_time is not None:
            conditions.append(
                {"key": "prep_time_minutes", "range": {"lte": max_prep_time}}
            )

        if prefer_traditional:
            conditions.append(
                {"key": "is_christmas_traditional", "match": {"value": True}}
            )

        if not conditions:
            return None

        if len(conditions) == 1:
            return conditions[0]

        return {"must": conditions}

    def search_recipes(
        self,
        query: str,
        category: RecipeCategory | None = None,
        n_results: int = 5,
        is_vegan: bool | None = None,
        is_vegetarian: bool | None = None,
        is_gluten_free: bool | None = None,
        is_dairy_free: bool | None = None,
        is_nut_free: bool | None = None,
        max_prep_time: int | None = None,
        prefer_traditional: bool = False,
    ) -> list[Recipe]:
        """
        Search for recipes with optional filters.

        Args:
            query: Search query text
            category: Filter by recipe category
            n_results: Maximum number of results to return
            is_vegan: Filter for vegan recipes
            is_vegetarian: Filter for vegetarian recipes
            is_gluten_free: Filter for gluten-free recipes
            is_dairy_free: Filter for dairy-free recipes
            is_nut_free: Filter for nut-free recipes
            max_prep_time: Maximum preparation time in minutes
            prefer_traditional: Prefer traditional Christmas recipes

        Returns:
            List of matching recipes
        """
        query_vector = self.embedder.embed(query)

        search_filter = self._build_filter(
            category=category,
            is_vegan=is_vegan,
            is_vegetarian=is_vegetarian,
            is_gluten_free=is_gluten_free,
            is_dairy_free=is_dairy_free,
            is_nut_free=is_nut_free,
            max_prep_time=max_prep_time,
            prefer_traditional=prefer_traditional,
        )

        try:
            results = self.vectorstore.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                k=n_results,
                filter=search_filter,
            )
        except Exception as e:
            print(f"Search error: {e}")
            results = self.vectorstore.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                k=n_results,
            )

        recipes = []
        for chunk in results:
            try:
                recipe_json = chunk.metadata.get("recipe_json")
                if recipe_json:
                    recipe = Recipe.model_validate_json(recipe_json)
                    recipes.append(recipe)
            except Exception as e:
                print(f"Error parsing recipe: {e}")
                continue

        return recipes

    def get_recipe_by_id(self, recipe_id: str) -> Recipe | None:
        """
        Get a specific recipe by ID.

        Args:
            recipe_id: The recipe ID

        Returns:
            Recipe if found, None otherwise
        """
        try:
            results = self.vectorstore.retrieve(
                collection_name=self.collection_name,
                ids=[recipe_id],
            )

            if results:
                recipe_json = results[0].metadata.get("recipe_json")
                if recipe_json:
                    return Recipe.model_validate_json(recipe_json)
        except Exception as e:
            print(f"Error getting recipe: {e}")

        return None

    def get_recipes_by_category(self, category: RecipeCategory) -> list[Recipe]:
        """
        Get all recipes in a specific category.

        Args:
            category: Recipe category

        Returns:
            List of recipes in the category
        """
        query_vector = self.embedder.embed("recipe")
        cat_value = category if isinstance(category, str) else category.value

        results = self.vectorstore.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            k=30,
            filter={"key": "category", "match": {"value": cat_value}},
        )

        recipes = []
        for chunk in results:
            try:
                recipe_json = chunk.metadata.get("recipe_json")
                if recipe_json:
                    recipe = Recipe.model_validate_json(recipe_json)
                    recipes.append(recipe)
            except Exception as e:
                print(f"Error parsing recipe: {e}")
                continue

        return recipes

    def count_recipes(self) -> int:
        """Get the total number of recipes in the store."""
        try:
            all_chunks = list(
                self.vectorstore.dump_collection(
                    collection_name=self.collection_name,
                    with_vectors=False,
                )
            )
            return len(all_chunks)
        except Exception:
            return 0

    def clear_all(self) -> None:
        """Clear all recipes from the store."""
        try:
            all_chunks = list(
                self.vectorstore.dump_collection(
                    collection_name=self.collection_name,
                    with_vectors=False,
                )
            )
            if all_chunks:
                ids = [chunk.id for chunk in all_chunks if chunk.id]
                if ids:
                    self.vectorstore.remove(
                        collection_name=self.collection_name, ids=ids
                    )
        except Exception as e:
            print(f"Error clearing recipes: {e}")
