from typing import Any
from datapizza.agents import Agent  # type: ignore

from ..config import create_client
from ..tools.recipe_search import (
    search_appetizers,
    search_main_dishes,
    search_second_plates,
    search_desserts,
    get_recipe_details,
)
from datapizza.tools.duckduckgo import DuckDuckGoSearchTool  # type: ignore

TRADITIONAL_DISHES: dict[str, Any] = {
    "appetizer": [],
    "main_dish": [],
    "second_plate": [],
    "dessert": [],
}  # TODO: ask directly the agent to see if they are traditional


class BaseRecipeAgent(Agent):
    """
    Base class for recipe search agents.
    Handles common logic for preferences, prompt building, and search.
    """

    CATEGORY = "recipe"
    SYSTEM_PROMPT = "You are a recipe search assistant."
    TOOLS = [get_recipe_details]  # Default tools, override in subclasses
    COURSE_NAME = "recipe"
    RECOMMENDED_COUNT = 2

    def __init__(
        self, api_key: str | None = None, provider: str | None = None, **kwargs
    ):
        """
        Initialize the recipe agent.

        Args:
            api_key: OpenAI API key (not needed for Ollama)
            provider: LLM provider override ("openai" or "ollama")
        """
        client = create_client(
            api_key=api_key,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.7,
            provider=provider,
        )

        super().__init__(
            name=f"{self.CATEGORY}_agent",
            client=client,
            system_prompt=self.SYSTEM_PROMPT,
            tools=self._get_tools(),
            max_steps=5,
            terminate_on_text=True,
            **kwargs,
        )

    def _get_tools(self) -> list:
        """Get the tools for this agent."""
        return self.TOOLS

    def _extract_preferences(self, preferences: dict[str, Any]) -> dict[str, Any]:
        allergies = preferences.get("allergies", [])
        custom_allergies = preferences.get("custom_allergies", [])
        if isinstance(allergies, str):
            if not isinstance(custom_allergies, str):
                if isinstance(custom_allergies, list):
                    custom_allergies = ', '.join(custom_allergies)
                else: 
                    custom_allergies = ""
        elif isinstance(allergies, list):
            if not isinstance(custom_allergies, list):
                if isinstance(custom_allergies, str):
                    custom_allergies = [custom_allergies]
                else: 
                    custom_allergies = []

        return {
            "has_vegans": preferences.get("has_vegans", False),
            "has_vegetarians": preferences.get("has_vegetarians", False),
            "prefer_traditional": preferences.get("prefer_traditional", True),
            "allergens": allergies + custom_allergies,
            "number_of_guests": preferences.get("number_of_guests", "unknown"),
            "vegan_count": preferences.get("vegan_count", 0),
            "vegetarian_count": preferences.get("vegetarian_count", 0),
        }

    def _build_prompt(self, prefs: dict[str, Any], context: str = "") -> str:
        traditional_options = TRADITIONAL_DISHES.get(self.CATEGORY, [])
        traditional_text = (
            f"Traditional options: {', '.join(traditional_options)}"
            if traditional_options
            else ""
        )
        return f"""
You are an expert Christmas {self.COURSE_NAME} specialist. Your role is to:

1. Search for {self.COURSE_NAME} recipes that match the user's preferences
2. Consider dietary restrictions (vegan, vegetarian, allergies)
3. Recommend {self.RECOMMENDED_COUNT} {self.COURSE_NAME} options
4. Prefer traditional recipes when requested
5. Provide brief explanations for each recommendation

{traditional_text}

Number of guests: {prefs["number_of_guests"]}
Vegan guests: {prefs["vegan_count"] if prefs["has_vegans"] else "None"}
Vegetarian guests: {prefs["vegetarian_count"] if prefs["has_vegetarians"] else "None"}
Allergies to avoid: {", ".join(prefs["allergens"]) if prefs["allergens"] else "None"}
Prefer traditional recipes: {"Yes" if prefs["prefer_traditional"] else "No"}

{f"Additional context: {context}" if context else ""}

Please search for {self.COURSE_NAME} and recommend {self.RECOMMENDED_COUNT} options.
Ensure vegan/vegetarian options are included if needed.
"""

    def search(self, preferences: dict[str, Any], context: str = "") -> dict[str, Any]:
        prefs = self._extract_preferences(preferences)
        prompt = self._build_prompt(prefs, context)
        response = self.run(prompt, tool_choice="required_first")
        return {
            "category": self.CATEGORY,
            "raw_response": getattr(response, "text", str(response)),
            "preferences_used": prefs,
        }


class AppetizerAgent(BaseRecipeAgent):
    """Agent specialized in searching for appetizer recipes."""

    CATEGORY = "appetizer"
    COURSE_NAME = "appetizer_expert"
    TOOLS = [search_appetizers, get_recipe_details]
    RECOMMENDED_COUNT = 3
    SYSTEM_PROMPT = """You are an expert Christmas appetizer specialist. Your role is to:

1. Search for appetizer recipes that match the user's preferences
2. Consider dietary restrictions (vegan, vegetarian, allergies)
3. Recommend 2-3 appetizers that complement each other
4. Prefer traditional Christmas appetizers when requested
5. Consider preparation time and difficulty

When searching:
- Use the search_appetizers tool with appropriate filters
- If vegan guests are present, ensure at least one vegan option
- If vegetarian guests are present, ensure at least one vegetarian option
- Balance light and rich appetizers

Output your recommendations with brief explanations for why each was chosen.
Format your response clearly with the recipe names, descriptions, and why they were selected."""


class MainDishAgent(BaseRecipeAgent):
    """Agent specialized in searching for main dish (primo) recipes."""

    CATEGORY = "main_dish"
    COURSE_NAME = "main_dish_expert"
    TOOLS = [search_main_dishes, get_recipe_details]
    RECOMMENDED_COUNT = 2
    SYSTEM_PROMPT = """You are an expert Christmas main dish (primo piatto) specialist. Your role is to:

1. Search for main dish recipes - typically pasta, risotto, or soup
2. Consider dietary restrictions (vegan, vegetarian, allergies)
3. Recommend 1-2 main dishes appropriate for a Christmas feast
4. Prefer traditional Christmas main dishes when requested
5. Consider the flow of the meal - this comes before the second plate

When searching:
- Use the search_main_dishes tool with appropriate filters
- If vegan guests are present, ensure at least one vegan option
- Traditional Italian Christmas main dishes include: tortellini in brodo, lasagna, pasta al forno
- Consider richness - main dish should not overpower the second plate

Output your recommendations with brief explanations for why each was chosen.
Format your response clearly with the recipe names, descriptions, and why they were selected."""


class SecondPlateAgent(BaseRecipeAgent):
    """Agent specialized in searching for second plate (secondo) recipes."""

    CATEGORY = "second_plate"
    COURSE_NAME = "second_plate_expert"
    TOOLS = [search_second_plates, get_recipe_details]
    RECOMMENDED_COUNT = 2
    SYSTEM_PROMPT = """You are an expert Christmas second plate (secondo piatto) specialist. Your role is to:

1. Search for second plate recipes - typically meat or fish dishes
2. Consider dietary restrictions (vegan, vegetarian, allergies)
3. Recommend 1-2 second plates appropriate for a Christmas feast
4. Prefer traditional Christmas second plates when requested
5. This is often the centerpiece of the meal

When searching:
- Use the search_second_plates tool with appropriate filters
- For vegan/vegetarian guests, suggest plant-based main courses
- Traditional options include: roasted turkey, beef tenderloin, baked fish
- Consider cooking time - some dishes need hours of preparation

Output your recommendations with brief explanations for why each was chosen.
Format your response clearly with the recipe names, descriptions, and why they were selected."""


class DessertAgent(BaseRecipeAgent):
    """Agent specialized in searching for dessert recipes."""

    CATEGORY = "dessert"
    COURSE_NAME = "dessert_expert"
    TOOLS = [search_desserts, get_recipe_details]
    RECOMMENDED_COUNT = 2
    SYSTEM_PROMPT = """You are an expert Christmas dessert specialist. Your role is to:

1. Search for dessert recipes that complete the Christmas feast
2. Consider dietary restrictions (vegan, vegetarian, allergies)
3. Recommend 1-2 desserts appropriate for Christmas
4. Prefer traditional Christmas desserts when requested
5. Balance richness with the rest of the meal

When searching:
- Use the search_desserts tool with appropriate filters
- Traditional Italian Christmas desserts: panettone, pandoro, tiramisù
- If guests have dietary restrictions, ensure alternatives exist
- Consider make-ahead desserts for easier meal preparation

Output your recommendations with brief explanations for why each was chosen.
Format your response clearly with the recipe names, descriptions, and why they were selected."""


class RecipeResearchAgent(Agent):
    """Agent for discovering new recipes from the web using DuckDuckGo search."""

    SYSTEM_PROMPT = """You are a recipe research expert. Your role is to:

1. Search the web for Christmas recipes when needed
2. Find recipes that match specific dietary requirements
3. Discover trending and popular Christmas dishes
4. Find alternatives for common allergens

When searching:
- Be specific in your search queries
- Look for recipes from reputable cooking sites
- Consider regional Christmas traditions
- Always verify dietary claims (vegan, gluten-free, etc.)

Output clear recipe suggestions with names, brief descriptions, and source URLs."""

    name = "recipe_researcher"

    def __init__(self, api_key: str | None = None, provider: str | None = None):
        """
        Initialize the recipe research agent with DuckDuckGo search.

        Args:
            api_key: OpenAI API key (not needed for Ollama)
            provider: LLM provider override ("openai" or "ollama")
        """
        client = create_client(
            api_key=api_key,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.7,
            provider=provider,
        )

        super().__init__(
            name="recipe_researcher",
            client=client,
            system_prompt=self.SYSTEM_PROMPT,
            tools=[DuckDuckGoSearchTool()],
            max_steps=5,
            terminate_on_text=True,
        )

    def search_web_recipes(self, query: str, dietary_requirements: str = "") -> str:
        """
        Search the web for recipes matching the query.

        Args:
            query: Recipe search query
            dietary_requirements: Optional dietary requirements to include

        Returns:
            Search results with recipe suggestions
        """
        search_prompt = f"""
Search for Christmas recipes matching: {query}
{f"Dietary requirements: {dietary_requirements}" if dietary_requirements else ""}

Find 2-3 suitable recipes and provide:
- Recipe name
- Brief description
- Key ingredients
- Why it's a good choice
"""
        response = self.run(search_prompt, tool_choice="required_first")
        return response.text if hasattr(response, "text") else str(response)
