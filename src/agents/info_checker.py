"""
Info Checker Agent - Validates and extracts user preferences for the Christmas menu.

Using datapizza-ai features:
- Memory for conversation context retention
- Native Agent class with stateless=False for state management
- max_steps and terminate_on_text for execution control

Supports both OpenAI and Ollama providers.
"""

import json
from typing import Optional
from datapizza.agents import Agent  #type: ignore
from datapizza.memory import Memory  #type: ignore
from datapizza.tools import tool  #type: ignore
from datapizza.type import ROLE, TextBlock  #type: ignore

from ..config import create_client
from ..models.user_preferences import UserPreferences, Allergy


# Stateless tools for the Info Checker Agent


@tool
def validate_preferences(preferences_json: str) -> str:
    """
    Validate the extracted user preferences.

    Args:
        preferences_json: JSON string containing user preferences

    Returns:
        Validation result with any missing information
    """
    try:
        data = json.loads(preferences_json)
        preferences = UserPreferences.model_validate(data)
    except Exception as e:
        return json.dumps(
            {
                "valid": False,
                "error": str(e),
                "message": "Failed to validate preferences",
            }
        )
    return json.dumps(
        {
            "valid": True,
            "preferences": preferences.model_dump(),
            "summary": preferences.get_dietary_requirements_summary(),
        }
    )


@tool
def create_preferences_object(
    number_of_guests: int,
    has_vegetarians: bool = False,
    vegetarian_count: int = 0,
    has_vegans: bool = False,
    vegan_count: int = 0,
    allergies: str = "",
    prefer_traditional: bool = True,
    max_difficulty: str = "medium",
    max_prep_time_minutes: Optional[int] = None,
    max_cook_time_minutes: Optional[int] = None,
    additional_notes: str = "",
) -> str:
    """
    Create a structured preferences object from extracted information.

    Args:
        number_of_guests: Total number of guests expected
        has_vegetarians: Whether there are vegetarian guests
        vegetarian_count: Number of vegetarian guests
        has_vegans: Whether there are vegan guests
        vegan_count: Number of vegan guests
        allergies: Comma-separated list of allergies (e.g., "nuts, dairy, gluten")
        prefer_traditional: Whether to prefer traditional Christmas recipes
        max_difficulty: Maximum recipe difficulty (easy, medium, hard)
        max_prep_time_minutes: Maximum preparation time per recipe
        additional_notes: Any additional notes or preferences

    Returns:
        JSON string with the structured preferences
    """
    allergy_list = []
    custom_allergies = []

    if allergies:
        for allergy in allergies.split(","):
            allergy = allergy.strip().lower()
            # Try to match with standard allergies
            try:
                allergy_list.append(Allergy(allergy))
            except ValueError:
                custom_allergies.append(allergy)

    preferences = UserPreferences(
        number_of_guests=number_of_guests,
        has_vegetarians=has_vegetarians,
        vegetarian_count=vegetarian_count,
        has_vegans=has_vegans,
        vegan_count=vegan_count,
        allergies=allergy_list,
        custom_allergies=custom_allergies,
        prefer_traditional=prefer_traditional,
        max_difficulty=max_difficulty,
        max_prep_time_minutes=max_prep_time_minutes,
        max_cook_time_minutes=max_cook_time_minutes,
        additional_notes=additional_notes if additional_notes else None,
    )

    return json.dumps(
        {
            "preferences": preferences.model_dump(),
            "summary": preferences.get_dietary_requirements_summary(),
        }
    )


def _create_finalize_extraction_tool(result_holder: dict):
    """
    Factory function that creates a finalize_extraction tool with captured state.

    This avoids using global variables by capturing the result_holder dict via closure.

    Args:
        result_holder: Dict that will be mutated to store the extraction result

    Returns:
        A tool function that captures the result_holder
    """

    @tool
    def finalize_extraction(
        is_complete: bool,
        preferences_json: str = "",
        missing_info: str = "",
        questions: str = "",
        summary: str = "",
    ) -> str:
        """
        Finalize and return the extraction result. ALWAYS call this tool at the end of extraction.

        Args:
            is_complete: Whether all required information has been gathered
            preferences_json: JSON string of the preferences object (from create_preferences_object)
            missing_info: Comma-separated list of missing fields (if incomplete)
            questions: Questions to ask the user for missing info (if incomplete)
            summary: Human-readable summary of the preferences

        Returns:
            Confirmation of the finalized result
        """
        preferences = None
        if preferences_json:
            try:
                data = json.loads(preferences_json)
                preferences = data.get("preferences", data)
            except json.JSONDecodeError:
                pass

        result_holder["result"] = {
            "is_complete": is_complete,
            "preferences": preferences,
            "missing_info": [m.strip() for m in missing_info.split(",") if m.strip()],
            "questions": [q.strip() for q in questions.split("|") if q.strip()],
            "summary": summary,
        }

        return json.dumps({"status": "success", "result": result_holder["result"]})

    return finalize_extraction


class InfoCheckerAgent(Agent):
    """
    Agent responsible for validating and extracting user preferences.
    Ensures all necessary information is collected before menu planning.

    Uses datapizza-ai Memory for conversation context retention,
    allowing iterative preference gathering across multiple interactions.

    Supports both OpenAI and Ollama providers.
    """

    SYSTEM_PROMPT = """You are an expert Christmas menu planning assistant. Your role is to gather all the information needed to plan the perfect Christmas menu.

## INFORMATION TO COLLECT

You need to collect the following information from the user:

**REQUIRED (must have before proceeding):**
- Number of guests (essential for portion planning)

**IMPORTANT (should ask if not mentioned):**
- Dietary restrictions: Are there vegetarian guests? How many? Are there vegan guests? How many?
- Food allergies: Does anyone have allergies? (nuts, dairy, gluten, shellfish, etc.)

**OPTIONAL (nice to have):**
- Preference for traditional vs modern recipes
- Time/difficulty constraints (easy, medium, hard recipes)
- Budget level (low, medium, high)
- Available kitchen equipment
- Any additional notes or special requests

## YOUR BEHAVIOR

1. **EXTRACT** any information the user has already provided
2. **IDENTIFY** what information is still missing
3. **ASK** friendly follow-up questions for the missing information
4. **DO NOT** assume information - always ask if unclear

## IMPORTANT GUIDELINES

- The NUMBER OF GUESTS is absolutely required - always ask if not provided
- Even if the user doesn't mention dietary restrictions or allergies, you MUST ask to confirm there are none
- Be conversational and friendly, not like a form
- Remember context from previous messages
- Ask 2-3 questions at a time maximum, don't overwhelm the user

## WORKFLOW

1. Use create_preferences_object to structure the information you have (use defaults for unknown fields)
2. Use validate_preferences to check completeness
3. ALWAYS call finalize_extraction at the end:
   - is_complete=True only if you have: number_of_guests AND confirmed dietary restrictions/allergies
   - is_complete=False if any essential info is missing
   - missing_info: comma-separated list of missing fields
   - questions: pipe-separated list of questions to ask the user
   - summary: what you know so far

4. In your text response, ALWAYS include your follow-up questions to the user if information is missing!

## EXAMPLE INTERACTION

User: "I need a Christmas menu"
You should:
- Call create_preferences_object with defaults
- Call finalize_extraction with is_complete=False, missing_info="number_of_guests,dietary_restrictions,allergies"
- In your response, ASK: "I'd be happy to help plan your Christmas menu! To get started, could you tell me:
  1. How many guests are you expecting?
  2. Are there any vegetarians or vegans in the group?
  3. Does anyone have food allergies I should know about?"

CRITICAL: You MUST call finalize_extraction at the end AND ask follow-up questions in your response if info is missing!"""

    name = "info_checker"

    def __init__(self, api_key: Optional[str] = None, provider: Optional[str] = None):
        """
        Initialize the Info Checker Agent with Memory support.

        Args:
            api_key: OpenAI API key (not needed for Ollama)
            provider: LLM provider override ("openai" or "ollama")
        """
        # Create client using factory (supports both OpenAI and Ollama)
        client = create_client(
            api_key=api_key,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.7,
            provider=provider,
        )

        # Initialize memory for conversation context
        self._conversation_memory = Memory()

        # Instance-level state for extraction results (replaces global variable)
        self._extraction_result_holder: dict = {"result": None}

        # Create the finalize_extraction tool with captured instance state
        self._finalize_extraction_tool = _create_finalize_extraction_tool(
            self._extraction_result_holder
        )

        super().__init__(
            name="info_checker",
            client=client,
            system_prompt=self.SYSTEM_PROMPT,
            tools=[
                validate_preferences,
                create_preferences_object,
                self._finalize_extraction_tool,
            ],
            memory=self._conversation_memory,
            max_steps=5,  # Limit execution steps
            terminate_on_text=False,  # Don't stop on text, wait for finalize_extraction
        )

    def _run_extraction(self, prompt: str, user_message: str) -> dict:
        """
        Common extraction logic shared by extract_preferences and update_preferences.

        Args:
            prompt: The prompt to send to the agent
            user_message: The user's original message (for memory)

        Returns:
            Dictionary with extraction results
        """
        # Clear previous extraction result
        self._extraction_result_holder["result"] = None

        # Run agent with memory context
        response = self.run(prompt, tool_choice="auto")
        response_text = response.text if hasattr(response, "text") else str(response)

        # Update memory with this interaction
        self._conversation_memory.add_turn(
            TextBlock(content=user_message), role=ROLE.USER
        )
        self._conversation_memory.add_turn(
            TextBlock(content=response_text), role=ROLE.ASSISTANT
        )

        # Get the structured result from the finalize_extraction tool
        extraction_result = self._extraction_result_holder.get("result")

        if extraction_result:
            return {
                "raw_response": response_text,
                **extraction_result,
            }

        # Fallback if agent didn't call finalize_extraction
        return {
            "raw_response": response_text,
            "is_complete": False,
            "preferences": None,
            "missing_info": [],
            "questions": [],
            "summary": "",
        }

    def extract_preferences(self, user_message: str) -> dict:
        """
        Extract and validate user preferences from their message.
        Uses Memory to maintain conversation context for iterative preference gathering.

        Args:
            user_message: The user's message describing their requirements

        Returns:
            Dictionary containing:
                - is_complete: Whether all required info is present
                - preferences: The UserPreferences object (if complete)
                - missing_info: List of missing fields (if incomplete)
                - questions: Follow-up questions to ask (if incomplete)
                - summary: Human-readable summary
        """
        prompt = f"""
Analyze the following user message and extract preferences for their Christmas menu.

User message: "{user_message}"

Steps:
1. Extract all information the user provided (guests, dietary needs, allergies, preferences)
2. Combine with any information from previous messages in our conversation
3. Use create_preferences_object with the extracted information
4. Use validate_preferences to check completeness
5. Call finalize_extraction with:
   - is_complete=True ONLY if you have number_of_guests AND confirmed dietary/allergy info
   - Include questions for any missing information
6. In your response, if information is missing, ASK the user friendly follow-up questions!

Remember: You need at minimum the number of guests, and should confirm dietary restrictions and allergies.
"""
        return self._run_extraction(prompt, user_message)

    def ask_missing_info(self, current_preferences: dict, missing_fields: list) -> str:
        """
        Generate questions to ask the user for missing information.
        Uses Memory context to avoid asking for already-provided information.

        Args:
            current_preferences: Currently known preferences
            missing_fields: List of fields that are missing

        Returns:
            String with friendly questions to ask the user
        """
        questions = []

        if "number_of_guests" in missing_fields:
            questions.append("How many guests are you expecting for Christmas dinner?")

        if not current_preferences.get(
            "has_vegetarians"
        ) and not current_preferences.get("has_vegans"):
            questions.append("Are there any vegetarian or vegan guests?")

        if not current_preferences.get("allergies"):
            questions.append("Does anyone have food allergies I should be aware of?")

        return (
            "\n".join(questions) if questions else "I have all the information I need!"
        )

    def update_preferences(self, additional_info: str) -> dict:
        """
        Update preferences with additional information from the user.
        This method leverages Memory to combine new info with existing context.

        Args:
            additional_info: New information from the user

        Returns:
            Updated preferences dictionary
        """
        prompt = f"""
The user has provided additional information: "{additional_info}"

Steps:
1. Combine this new information with what we already know from our conversation
2. Use create_preferences_object with all the combined information
3. Use validate_preferences to check if we now have everything needed
4. Call finalize_extraction with:
   - is_complete=True ONLY if you have number_of_guests AND confirmed dietary/allergy info
   - Include questions for any still-missing information
5. In your response, if any information is still missing, ASK the user friendly follow-up questions!

Remember: You need at minimum the number of guests, and should confirm dietary restrictions and allergies.
"""
        return self._run_extraction(prompt, additional_info)

    def clear_memory(self) -> None:
        """Clear the conversation memory for a fresh start."""
        self._conversation_memory = Memory()
        self.memory = self._conversation_memory
        self._extraction_result_holder["result"] = None

    def get_memory_summary(self) -> str:
        """Get a summary of the current conversation context."""
        try:
            return self._conversation_memory.json_dumps()
        except Exception:
            return "No memory content available"

