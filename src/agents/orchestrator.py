from typing import Any, Generator, AsyncGenerator

from loguru import logger

from ..config import settings, get_provider_name, get_model_name
from ..agents.info_checker import InfoCheckerAgent
from ..agents.recipe_agents import (
    AppetizerAgent,
    MainDishAgent,
    RecipeResearchAgent,
    SecondPlateAgent,
    DessertAgent,
)
from ..agents.menu_creator import MenuCreatorAgent
from ..database.vector_store import RecipeVectorStore
from ..database.recipe_loader import RecipeLoader


class ChristmasMenuOrchestrator:
    """
    Main orchestrator that coordinates all agents in the Christmas menu planning pipeline.

    Uses datapizza-ai native multi-agent features:
    - can_call() for agent-to-agent communication
    - stream_invoke() for real-time progress updates
    - a_stream_invoke() for async streaming
    - Memory for conversation context
    - DuckDuckGoSearchTool for web recipe discovery

    Supports both OpenAI and Ollama providers.

    Pipeline:
    1. Info Checker Agent - Validates user requirements (with Memory)
    2. Recipe Research Agent - Web search for additional recipes (optional)
    3. Recipe Search Agents - Search for recipes in each category
    4. Menu Creator Agent - Compiles the final menu (with planning_interval)
    """

    def __init__(
        self,
        api_key: str | None = None,
        provider: str | None = None,
        initialize_db: bool = True,
    ):
        """
        Initialize the orchestrator with all agents.

        Args:
            api_key: OpenAI API key (not needed for Ollama)
            provider: LLM provider override ("openai" or "ollama")
            initialize_db: Whether to initialize the vector database with sample recipes.
        """
        self.api_key = api_key
        self.provider = provider or settings.LLM_PROVIDER

        self.info_checker = InfoCheckerAgent(
            api_key=self.api_key, provider=self.provider
        )
        self.appetizer_agent = AppetizerAgent(
            api_key=self.api_key, provider=self.provider
        )
        self.main_dish_agent = MainDishAgent(
            api_key=self.api_key, provider=self.provider
        )
        self.second_plate_agent = SecondPlateAgent(
            api_key=self.api_key, provider=self.provider
        )
        self.dessert_agent = DessertAgent(api_key=self.api_key, provider=self.provider)

        self.recipe_researcher = RecipeResearchAgent(
            api_key=self.api_key, provider=self.provider
        )

        self.menu_creator = MenuCreatorAgent(
            api_key=self.api_key, provider=self.provider
        )

        self.menu_creator.connect_recipe_agents(
            [
                self.appetizer_agent,
                self.main_dish_agent,
                self.second_plate_agent,
                self.dessert_agent,
                self.recipe_researcher,
            ]
        )

        self.vector_store = RecipeVectorStore()

        if initialize_db:
            self._ensure_recipes_loaded()

    def _ensure_recipes_loaded(self) -> None:
        """Ensure the vector database has recipes loaded."""
        if self.vector_store.count_recipes() == 0:
            logger.info("📚 Loading sample recipes into database...")
            loader = RecipeLoader(self.vector_store)
            count = loader.load_sample_recipes()
            logger.info(f"✅ Loaded {count} sample recipes")

    def get_provider_info(self) -> str:
        """Get information about the current LLM provider."""
        return f"{get_provider_name()} ({get_model_name()})"

    def run(self, user_request: str, interactive: bool = False) -> dict[str, Any]:
        """
        Run the complete menu planning pipeline.

        Args:
            user_request: The user's request for a Christmas menu
            interactive: If True, will prompt for missing information

        Returns:
            Dictionary containing:
                - success: Whether the pipeline completed successfully
                - menu: The final formatted menu (if successful)
                - preferences: The extracted user preferences
                - agent_outputs: Individual outputs from each agent
                - error: Error message (if failed)
        """
        logger.info("\n🎄 Christmas Menu Planner 🎄")
        logger.info(f"Using: {self.get_provider_info()}")
        logger.info("=" * 50)

        result: dict[str, Any] = {
            "success": False,
            "menu": None,
            "preferences": None,
            "agent_outputs": {},
            "error": None,
            "provider": self.get_provider_info(),
        }

        logger.info("\n📋 Step 1: Analyzing your requirements...")
        preferences_result = self.info_checker.extract_preferences(user_request)
        result["agent_outputs"]["info_checker"] = preferences_result

        if not preferences_result.get("is_complete"):
            if interactive:
                logger.warning("⚠️  Some information is missing. Please provide:")
                questions = self.info_checker.ask_missing_info(
                    preferences_result.get("preferences", {}),
                    preferences_result.get("missing_info", []),
                )
                logger.info(questions)
                result["error"] = "Missing required information"
                return result
            else:
                logger.warning("⚠️  Using defaults for missing information")
                preferences = preferences_result.get("preferences") or {
                    "number_of_guests": 6,
                    "has_vegetarians": False,
                    "vegetarian_count": 0,
                    "has_vegans": False,
                    "vegan_count": 0,
                    "allergies": [],
                    "custom_allergies": [],
                    "prefer_traditional": True,
                    "max_difficulty": "medium",
                }
        else:
            preferences = preferences_result.get("preferences", {})

        result["preferences"] = preferences
        logger.info(
            f"✅ Preferences extracted: {preferences.get('number_of_guests', 'N/A')} guests"
        )

        # Step 2: Search for recipes using specialized agents
        logger.info("\n🔍 Step 2: Searching for recipes...")
        logger.info("   • Searching appetizers...")
        appetizer_result = self.appetizer_agent.search(preferences)
        result["agent_outputs"]["appetizer"] = appetizer_result

        logger.info("   • Searching main dishes...")
        main_dish_result = self.main_dish_agent.search(preferences)
        result["agent_outputs"]["main_dish"] = main_dish_result

        logger.info("   • Searching second plates...")
        second_plate_result = self.second_plate_agent.search(preferences)
        result["agent_outputs"]["second_plate"] = second_plate_result

        logger.info("   • Searching desserts...")
        dessert_result = self.dessert_agent.search(preferences)
        result["agent_outputs"]["dessert"] = dessert_result

        logger.info("✅ Recipe search complete!")

        # Step 3: Create the final menu using planning_interval
        logger.info("\n📝 Step 3: Creating your personalized menu...")
        menu_result = self.menu_creator.create_menu(
            preferences=preferences,
            appetizer_suggestions=appetizer_result.get("raw_response", ""),
            main_dish_suggestions=main_dish_result.get("raw_response", ""),
            second_plate_suggestions=second_plate_result.get("raw_response", ""),
            dessert_suggestions=dessert_result.get("raw_response", ""),
        )

        result["agent_outputs"]["menu_creator"] = menu_result
        result["menu"] = menu_result.get("formatted_menu", "")
        result["success"] = True

        logger.info("✅ Menu created successfully!")
        logger.info("\n" + "=" * 50)

        return result

    def run_streaming(self, user_request: str) -> Generator[dict[str, Any], None, None]:
        """
        Run the pipeline with streaming progress updates.
        Uses stream_invoke() for real-time feedback.

        Args:
            user_request: The user's request for a Christmas menu

        Yields:
            Progress updates at each step
        """
        yield {
            "type": "status",
            "message": f"🎄 Starting Christmas Menu Planner ({self.get_provider_info()})...",
        }

        # Step 1: Extract preferences
        yield {"type": "status", "message": "📋 Analyzing your requirements..."}
        preferences_result = self.info_checker.extract_preferences(user_request)

        if not preferences_result.get("is_complete"):
            preferences = preferences_result.get("preferences") or {
                "number_of_guests": 6,
                "has_vegetarians": False,
                "vegetarian_count": 0,
                "has_vegans": False,
                "vegan_count": 0,
                "allergies": [],
                "custom_allergies": [],
                "prefer_traditional": True,
                "max_difficulty": "medium",
            }
            yield {
                "type": "warning",
                "message": "⚠️ Using defaults for missing information",
            }
        else:
            preferences = preferences_result.get("preferences", {})

        yield {
            "type": "progress",
            "step": "info_checker",
            "message": f"✅ Preferences extracted: {preferences.get('number_of_guests', 'N/A')} guests",
            "data": preferences,
        }

        # Step 2: Search recipes with streaming progress
        recipe_agents = [
            ("appetizer", self.appetizer_agent, "🥗 Searching appetizers..."),
            ("main_dish", self.main_dish_agent, "🍝 Searching main dishes..."),
            ("second_plate", self.second_plate_agent, "🥩 Searching second plates..."),
            ("dessert", self.dessert_agent, "🍰 Searching desserts..."),
        ]

        recipe_results = {}
        for category, agent, message in recipe_agents:
            yield {"type": "status", "message": message}
            result = agent.search(preferences)
            recipe_results[category] = result
            yield {
                "type": "progress",
                "step": category,
                "message": f"✅ Found {category.replace('_', ' ')} suggestions",
                "data": result,
            }

        # Step 3: Create menu with streaming
        yield {"type": "status", "message": "📝 Creating your personalized menu..."}

        # Stream menu creation steps
        final_menu = None
        for step in self.menu_creator.create_menu_streaming(
            preferences=preferences,
            appetizer_suggestions=recipe_results["appetizer"].get("raw_response", ""),
            main_dish_suggestions=recipe_results["main_dish"].get("raw_response", ""),
            second_plate_suggestions=recipe_results["second_plate"].get(
                "raw_response", ""
            ),
            dessert_suggestions=recipe_results["dessert"].get("raw_response", ""),
        ):
            step_text = step.text if hasattr(step, "text") else str(step)
            yield {
                "type": "step",
                "step_index": step.index if hasattr(step, "index") else 0,
                "message": step_text[:100] + "..."
                if len(step_text) > 100
                else step_text,
            }
            final_menu = step_text

        yield {
            "type": "complete",
            "message": "✅ Menu created successfully!",
            "menu": final_menu,
            "preferences": preferences,
        }

    async def run_async(self, user_request: str) -> dict[str, Any]:
        """
        Run the pipeline asynchronously using true async execution.

        Args:
            user_request: The user's request for a Christmas menu

        Returns:
            Same as run()
        """
        result: dict[str, Any] = {
            "success": False,
            "menu": None,
            "preferences": None,
            "agent_outputs": {},
            "error": None,
            "provider": self.get_provider_info(),
        }

        # Step 1: Extract preferences
        preferences_result = self.info_checker.extract_preferences(user_request)
        result["agent_outputs"]["info_checker"] = preferences_result

        if not preferences_result.get("is_complete"):
            preferences = preferences_result.get("preferences") or {
                "number_of_guests": 6,
                "has_vegetarians": False,
                "vegetarian_count": 0,
                "has_vegans": False,
                "vegan_count": 0,
                "allergies": [],
                "custom_allergies": [],
                "prefer_traditional": True,
                "max_difficulty": "medium",
            }
        else:
            preferences = preferences_result.get("preferences", {})

        result["preferences"] = preferences

        # Step 2: Search recipes (could run in parallel with asyncio.gather if agents support async)
        result["agent_outputs"]["appetizer"] = self.appetizer_agent.search(preferences)
        result["agent_outputs"]["main_dish"] = self.main_dish_agent.search(preferences)
        result["agent_outputs"]["second_plate"] = self.second_plate_agent.search(
            preferences
        )
        result["agent_outputs"]["dessert"] = self.dessert_agent.search(preferences)

        # Step 3: Create menu
        menu_result = self.menu_creator.create_menu(
            preferences=preferences,
            appetizer_suggestions=result["agent_outputs"]["appetizer"].get(
                "raw_response", ""
            ),
            main_dish_suggestions=result["agent_outputs"]["main_dish"].get(
                "raw_response", ""
            ),
            second_plate_suggestions=result["agent_outputs"]["second_plate"].get(
                "raw_response", ""
            ),
            dessert_suggestions=result["agent_outputs"]["dessert"].get(
                "raw_response", ""
            ),
        )

        result["agent_outputs"]["menu_creator"] = menu_result
        result["menu"] = menu_result.get("formatted_menu", "")
        result["success"] = True

        return result

    async def run_async_streaming(
        self, user_request: str
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Run the pipeline with async streaming using a_stream_invoke().

        Args:
            user_request: The user's request for a Christmas menu

        Yields:
            Progress updates at each step
        """
        yield {
            "type": "status",
            "message": f"🎄 Starting Christmas Menu Planner (Async - {self.get_provider_info()})...",
        }

        # Step 1: Extract preferences
        yield {"type": "status", "message": "📋 Analyzing your requirements..."}
        preferences_result = self.info_checker.extract_preferences(user_request)

        if not preferences_result.get("is_complete"):
            preferences = preferences_result.get("preferences") or {
                "number_of_guests": 6,
                "has_vegetarians": False,
                "vegetarian_count": 0,
                "has_vegans": False,
                "vegan_count": 0,
                "allergies": [],
                "custom_allergies": [],
                "prefer_traditional": True,
                "max_difficulty": "medium",
            }
        else:
            preferences = preferences_result.get("preferences", {})

        yield {
            "type": "progress",
            "step": "info_checker",
            "message": "✅ Preferences extracted",
            "data": preferences,
        }

        # Step 2: Search recipes
        recipe_results = {}
        for category, agent in [
            ("appetizer", self.appetizer_agent),
            ("main_dish", self.main_dish_agent),
            ("second_plate", self.second_plate_agent),
            ("dessert", self.dessert_agent),
        ]:
            yield {
                "type": "status",
                "message": f"🔍 Searching {category.replace('_', ' ')}...",
            }
            result = agent.search(preferences)
            recipe_results[category] = result
            yield {"type": "progress", "step": category, "data": result}

        # Step 3: Create menu with async streaming
        yield {"type": "status", "message": "📝 Creating menu..."}

        final_menu = None
        async for step in self.menu_creator.a_stream_invoke(
            f"Create Christmas menu for {preferences.get('number_of_guests', 6)} guests"
        ):
            step_text = step.text if hasattr(step, "text") else str(step)
            yield {"type": "step", "message": step_text}
            final_menu = step_text

        yield {"type": "complete", "menu": final_menu, "preferences": preferences}

    def run_with_native_agents(self, user_request: str) -> dict[str, Any]:
        """
        Run the pipeline using native can_call() multi-agent collaboration.
        The menu creator directly calls recipe agents as tools.

        Args:
            user_request: The user's request for a Christmas menu

        Returns:
            Dictionary with menu and metadata
        """
        logger.info(
            f"\n🎄 Christmas Menu Planner (Native Multi-Agent - {self.get_provider_info()}) 🎄"
        )
        logger.info("=" * 50)

        # Step 1: Extract preferences
        logger.info("\n📋 Step 1: Analyzing your requirements...")
        preferences_result = self.info_checker.extract_preferences(user_request)

        if not preferences_result.get("is_complete"):
            preferences = preferences_result.get("preferences") or {
                "number_of_guests": 6,
                "has_vegetarians": False,
                "vegetarian_count": 0,
                "has_vegans": False,
                "vegan_count": 0,
                "allergies": [],
                "custom_allergies": [],
                "prefer_traditional": True,
                "max_difficulty": "medium",
            }
        else:
            preferences = preferences_result.get("preferences", {})

        logger.info(f"✅ Preferences: {preferences.get('number_of_guests', 'N/A')} guests")

        # Step 2: Let menu creator handle everything via can_call()
        logger.info("\n🤖 Step 2: Menu Creator coordinating with recipe experts...")
        menu_result = self.menu_creator.create_menu_with_agents(preferences)

        logger.info("✅ Menu created successfully!")
        logger.info("\n" + "=" * 50)

        return {
            "success": True,
            "menu": menu_result.get("formatted_menu", ""),
            "preferences": preferences,
            "provider": self.get_provider_info(),
        }

    def search_web_recipes(self, query: str, dietary_requirements: str = "") -> str:
        """
        Search the web for recipes using DuckDuckGo.

        Args:
            query: Recipe search query
            dietary_requirements: Optional dietary requirements

        Returns:
            Web search results
        """
        return self.recipe_researcher.search_web_recipes(query, dietary_requirements)

    def get_recipe_count(self) -> int:
        """Get the number of recipes in the database."""
        return self.vector_store.count_recipes()

    def reload_recipes(self, json_file: str | None = None) -> int:
        """
        Reload recipes into the database.

        Args:
            json_file: Path to JSON file with recipes. Uses sample recipes if None.

        Returns:
            Number of recipes loaded
        """
        self.vector_store.clear_all()
        loader = RecipeLoader(self.vector_store)

        if json_file:
            return loader.load_from_json_file(json_file)
        else:
            return loader.load_sample_recipes()

    def clear_info_checker_memory(self) -> None:
        """Clear the info checker's conversation memory."""
        self.info_checker.clear_memory()


def create_orchestrator(
    api_key: str | None = None,
    provider: str | None = None,
) -> ChristmasMenuOrchestrator:
    """
    Factory function to create an orchestrator instance.

    Args:
        api_key: OpenAI API key (not needed for Ollama)
        provider: LLM provider ("openai" or "ollama")

    Returns:
        Configured ChristmasMenuOrchestrator instance
    """
    return ChristmasMenuOrchestrator(api_key=api_key, provider=provider)
