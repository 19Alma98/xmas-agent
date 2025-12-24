"""
Document Parsing Agent - Extracts recipe information from PDF documents.

Uses datapizza AI Agent to intelligently extract recipe information from PDF text
and structure it for storage in the vector database.
"""

import json
import re
from pydantic import BaseModel, Field
from datapizza.agents import Agent  # type: ignore

from ..config import create_client


class ExtractedRecipe(BaseModel):
    """Pydantic model for extracted recipe data."""

    name: str
    description: str
    category: str = Field(
        description="One of: appetizer, main_dish, second_plate, dessert"
    )
    ingredients: list[str] = Field(default_factory=list)
    instructions: list[str] = Field(default_factory=list)
    servings: int = Field(default=4)
    prep_time_minutes: int | None = None
    cook_time_minutes: int | None = None
    dietary_tags: list[str] = Field(default_factory=list)
    allergens: list[str] = Field(default_factory=list)
    difficulty: str = Field(default="medium", description="One of: easy, medium, hard")
    is_christmas_traditional: bool = Field(default=True)


class RecipeList(BaseModel):
    """Wrapper model for a list of recipes."""

    recipes: list[ExtractedRecipe]


class DocumentParsingAgent(Agent):
    """
    Agent that extracts recipe information from document text.
    Uses AI to intelligently parse and structure recipe data.
    """

    SYSTEM_PROMPT = """You are an expert at extracting recipe information from text.
Your task is to analyze text from PDF documents and extract structured recipe information.

For each recipe you find, extract:
- Recipe name
- Description
- Category (appetizer, main_dish, second_plate, or dessert)
- Ingredients list
- Instructions
- Estimated servings
- Estimated prep time and cook time (if mentioned)
- Dietary tags (vegan, vegetarian, gluten_free, dairy_free, nut_free)
- Allergens
- Difficulty level (easy, medium, hard)
- Whether it's a traditional Christmas recipe

Return the information as a JSON array of recipe objects."""

    def __init__(
        self, api_key: str | None = None, provider: str | None = None, **kwargs
    ):
        """
        Initialize the document parsing agent.

        Args:
            api_key: OpenAI API key (not needed for Ollama)
            provider: LLM provider override ("openai" or "ollama")
        """
        client = create_client(
            api_key=api_key,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.1,
            provider=provider,
        )

        super().__init__(
            name="document_parsing_agent",
            client=client,
            system_prompt=self.SYSTEM_PROMPT,
            tools=[],
            max_steps=3,
            terminate_on_text=True,
            **kwargs,
        )

    def extract_recipes(self, text: str) -> list[dict]:
        """
        Extract recipe information from text using AI.
        Uses structured response when possible, falls back to JSON extraction.

        Args:
            text: Text content from PDF document

        Returns:
            List of recipe dictionaries with extracted information
        """
        prompt = f"""Extract all recipe information from the following text.
Return a JSON array of recipe objects. Each recipe should have:
- name: string
- description: string
- category: one of "appetizer", "main_dish", "second_plate", "dessert"
- ingredients: array of strings
- instructions: array of strings
- servings: integer (default to 4 if not specified)
- prep_time_minutes: integer or null
- cook_time_minutes: integer or null
- dietary_tags: array of strings (e.g., ["vegetarian", "vegan"])
- allergens: array of strings
- difficulty: "easy", "medium", or "hard"
- is_christmas_traditional: boolean

Text to analyze:
{text}
"""

        try:
            if hasattr(self.client, "structured_response"):
                try:
                    response = self.client.structured_response(
                        input=prompt, output_cls=RecipeList
                    )
                    if response.structured_data and len(response.structured_data) > 0:
                        recipe_list = response.structured_data[0]
                        return [recipe.model_dump() for recipe in recipe_list.recipes]
                except Exception as e:
                    print(
                        f"Structured response failed, falling back to JSON extraction: {e}"
                    )

            response = self.invoke(prompt)
            response_text = (
                response.text if hasattr(response, "text") else str(response)
            )

            recipes = self._extract_json_from_response(response_text)
            return recipes

        except Exception as e:
            print(f"Error extracting recipes: {e}")
            return []

    def _extract_json_from_response(self, response_text: str) -> list[dict]:
        """
        Extract JSON from agent response, handling various formats:
        - Markdown code blocks (```json ... ```)
        - Plain JSON arrays/objects
        - JSON wrapped in text

        Args:
            response_text: Raw response text from agent

        Returns:
            List of recipe dictionaries
        """
        json_block_pattern = r"```(?:json)?\s*(\[[\s\S]*?\])\s*```"
        match = re.search(json_block_pattern, response_text, re.IGNORECASE)
        if match:
            try:
                recipes = json.loads(match.group(1))
                return recipes if isinstance(recipes, list) else [recipes]
            except json.JSONDecodeError:
                pass

        json_start = response_text.find("[")
        json_end = response_text.rfind("]") + 1

        if json_start != -1 and json_end > json_start:
            try:
                json_str = response_text[json_start:json_end]
                recipes = json.loads(json_str)
                return recipes if isinstance(recipes, list) else [recipes]
            except json.JSONDecodeError:
                pass
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1

        if json_start != -1 and json_end > json_start:
            try:
                json_str = response_text[json_start:json_end]
                recipe = json.loads(json_str)
                return [recipe] if isinstance(recipe, dict) else []
            except json.JSONDecodeError:
                pass

        print(
            f"Warning: Could not extract JSON from agent response: {response_text[:200]}"
        )
        return []
