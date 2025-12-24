from typing import Any, Generator
from datapizza.agents import Agent  # type: ignore
from datapizza.tools import tool  # type: ignore

from ..config import create_client


@tool
def format_final_menu(
    title: str,
    appetizers: str,
    main_dishes: str,
    second_plates: str,
    desserts: str,
    number_of_guests: int,
    preparation_notes: str,
    shopping_tips: str,
) -> str:
    """
    Format the final Christmas menu in a beautiful presentation format.

    Args:
        title: Title for the menu (e.g., "Christmas Dinner 2025")
        appetizers: List of selected appetizers with descriptions
        main_dishes: List of selected main dishes with descriptions
        second_plates: List of selected second plates with descriptions
        desserts: List of selected desserts with descriptions
        number_of_guests: Number of guests the menu serves
        preparation_notes: Optional notes on preparation timeline
        shopping_tips: Optional shopping and preparation tips

    Returns:
        Beautifully formatted menu string
    """
    menu = f"""
╔══════════════════════════════════════════════════════════════════╗
║                    🎄 {title.center(42)} 🎄                    ║
║                      For {number_of_guests} Guests                              ║
╚══════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🥗 APPETIZERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{appetizers}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🍝 MAIN DISHES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{main_dishes}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🥩 SECOND PLATES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{second_plates}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🍰 DESSERTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{desserts}
"""

    if preparation_notes:
        menu += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 PREPARATION TIMELINE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{preparation_notes}
"""

    if shopping_tips:
        menu += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🛒 SHOPPING TIPS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{shopping_tips}
"""

    menu += """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                     🎅 Happy Christmas! 🎅
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

    return menu


class MenuCreatorAgent(Agent):
    """
    Agent responsible for compiling recipe suggestions into a cohesive menu.
    Takes input from all recipe agents and creates the final Christmas menu.
    """

    SYSTEM_PROMPT = """You are an expert Christmas menu curator and food coordinator. Your role is to:

1. REVIEW recipe suggestions from the specialized agents (appetizers, main dishes, second plates, desserts)
2. SELECT the best combination that creates a balanced and harmonious menu
3. ENSURE dietary requirements are met for all guests
4. SUGGEST wine pairings that complement the dishes
5. PROVIDE a preparation timeline for the cook
6. FORMAT the final menu beautifully

Menu Creation Guidelines:
- Balance flavors: don't repeat similar tastes across courses
- Balance richness: alternate lighter and richer dishes
- Consider preparation complexity: spread difficult dishes across the day
- Respect tradition while accommodating modern dietary needs
- Ensure vegan/vegetarian guests have satisfying options for EVERY course

Planning Steps:
1. First, review all available recipe options from each category
2. Identify dietary requirements and constraints
3. Select recipes that work well together as a complete meal
4. Plan wine pairings and preparation timeline
5. Format the final menu using the format_final_menu tool

Output:
Use the format_final_menu tool to create a beautiful, printable menu.
Include wine pairings and a preparation timeline when possible."""

    PLANNING_PROMPT = """Review your progress in creating the Christmas menu.

Consider:
- What recipe categories have been covered?
- Are all dietary requirements being met?
- Do the selected dishes complement each other?
- What's left to finalize?

Plan your next steps to complete the menu creation."""

    name = "menu_creator"

    def __init__(
        self,
        api_key: str | None = None,
        provider: str | None = None,
        recipe_agents: list[Agent] | None = None,
    ):
        """
        Initialize the Menu Creator Agent.

        Args:
            api_key: OpenAI API key (not needed for Ollama)
            provider: LLM provider override ("openai" or "ollama")
        """
        client = create_client(
            api_key=api_key,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.8,  # creative menu curation
            provider=provider,
        )

        super().__init__(
            name=self.name,
            client=client,
            system_prompt=self.SYSTEM_PROMPT,
            tools=[format_final_menu],
            max_steps=10,
            terminate_on_text=True,
            planning_interval=3,
            planning_prompt=self.PLANNING_PROMPT,
        )

        if recipe_agents:
            self.can_call(recipe_agents)

    def connect_recipe_agents(self, recipe_agents: list[Agent]) -> None:
        """
        Connect recipe agents for native multi-agent collaboration.

        Args:
            recipe_agents: List of specialized recipe agents
        """
        self.can_call(recipe_agents)

    def _extract_preferences(self, preferences: dict[str, Any]) -> dict[str, Any]:
        """Centralized extraction of guest preferences and dietary info."""
        number_of_guests = preferences.get("number_of_guests")
        if number_of_guests is None:
            raise ValueError("Number of guests is mandatory!")

        has_vegans = preferences.get("has_vegans", False)
        has_vegetarians = preferences.get("has_vegetarians", False)
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
        allergens = allergies + custom_allergies

        return {
            "number_of_guests": number_of_guests,
            "has_vegans": has_vegans,
            "has_vegetarians": has_vegetarians,
            "allergens": allergens,
        }

    def _build_menu_prompt(
        self,
        preferences: dict[str, Any],
        appetizer_suggestions: str = "",
        main_dish_suggestions: str = "",
        second_plate_suggestions: str = "",
        dessert_suggestions: str = "",
        traditional: bool | None = None,
    ) -> str:
        """Centralized prompt builder to avoid repetition."""
        dietary_info = self._extract_preferences(preferences)
        return f"""
Create a complete Christmas dinner menu:

GUEST INFORMATION:
- Number of guests: {dietary_info["number_of_guests"]}
- Vegan guests: {preferences.get("vegan_count", 0) if dietary_info["has_vegans"] else "None"}
- Vegetarian guests: {preferences.get("vegetarian_count", 0) if dietary_info["has_vegetarians"] else "None"}
- Allergies to avoid: {", ".join(dietary_info["allergens"]) if dietary_info["allergens"] else "None"}
- Prefer traditional: {"Yes" if traditional or preferences.get("prefer_traditional", True) else "No"}

APPETIZER SUGGESTIONS:
{appetizer_suggestions}

MAIN DISH SUGGESTIONS:
{main_dish_suggestions}

SECOND PLATE SUGGESTIONS:
{second_plate_suggestions}

DESSERT SUGGESTIONS:
{dessert_suggestions}

Instructions:
1. Select best recipes from each category
2. Ensure dietary requirements
3. Suggest wine pairings
4. Create preparation timeline
5. Use format_final_menu tool to create beautiful menu
"""

    def create_menu(
        self,
        preferences: dict[str, Any],
        appetizer_suggestions: str,
        main_dish_suggestions: str,
        second_plate_suggestions: str,
        dessert_suggestions: str,
    ) -> dict[str, Any]:
        """
        Create the final Christmas menu from all recipe suggestions.

        Args:
            preferences: User preferences dictionary
            appetizer_suggestions: Raw output from AppetizerAgent
            main_dish_suggestions: Raw output from MainDishAgent
            second_plate_suggestions: Raw output from SecondPlateAgent
            dessert_suggestions: Raw output from DessertAgent

        Returns:
            Dictionary with the formatted menu and metadata
        """
        prompt = self._build_menu_prompt(
            preferences,
            appetizer_suggestions,
            main_dish_suggestions,
            second_plate_suggestions,
            dessert_suggestions,
        )

        response = self.run(prompt, tool_choice="auto")
        response_text = response.text if hasattr(response, "text") else str(response)

        dietary_info = self._extract_preferences(preferences)
        return {
            "formatted_menu": response_text,
            "number_of_guests": dietary_info["number_of_guests"],
            "dietary_accommodations": {
                "vegan_options": dietary_info["has_vegans"],
                "vegetarian_options": dietary_info["has_vegetarians"],
                "allergens_avoided": dietary_info["allergens"],
            },
        }

    def create_menu_streaming(
        self,
        preferences: dict[str, Any],
        appetizer_suggestions: str,
        main_dish_suggestions: str,
        second_plate_suggestions: str,
        dessert_suggestions: str,
    ) -> Generator:
        """
        Create the final Christmas menu with streaming progress.
        Yields step-by-step progress during menu creation.

        Args:
            preferences: User preferences dictionary
            appetizer_suggestions: Raw output from AppetizerAgent
            main_dish_suggestions: Raw output from MainDishAgent
            second_plate_suggestions: Raw output from SecondPlateAgent
            dessert_suggestions: Raw output from DessertAgent

        Yields:
            StepResult objects with progress information
        """
        prompt = self._build_menu_prompt(
            preferences,
            appetizer_suggestions,
            main_dish_suggestions,
            second_plate_suggestions,
            dessert_suggestions,
        )

        for step in self.stream_invoke(prompt):
            yield step

    def create_menu_with_agents(self, preferences: dict[str, Any]) -> dict[str, Any]:
        """
        Create menu by directly calling connected recipe agents via can_call().
        This demonstrates native multi-agent collaboration.

        Args:
            preferences: User preferences dictionary

        Returns:
            Dictionary with the formatted menu and metadata
        """
        prompt = self._build_menu_prompt(preferences, traditional=True)
        response = self.run(prompt, tool_choice="auto")
        response_text = getattr(response, "text", str(response))

        dietary_info = self._extract_preferences(preferences)
        return {
            "formatted_menu": response_text,
            "number_of_guests": dietary_info["number_of_guests"],
            "dietary_accommodations": {
                "vegan_options": dietary_info["has_vegans"],
                "vegetarian_options": dietary_info["has_vegetarians"],
                "allergens_avoided": dietary_info["allergens"],
            },
        }
