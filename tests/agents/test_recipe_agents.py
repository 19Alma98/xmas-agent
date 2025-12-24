from unittest.mock import Mock, patch

from src.agents.recipe_agents import (
    AppetizerAgent,
    MainDishAgent,
    SecondPlateAgent,
    DessertAgent,
    RecipeResearchAgent,
)


def test_appetizer_agent_is_correctly_initialized():
    """Test that AppetizerAgent is correctly initialized."""
    agent = AppetizerAgent()
    assert agent.name == "appetizer_agent"
    assert agent.CATEGORY == "appetizer"
    assert agent.COURSE_NAME == "appetizer_expert"
    assert agent.RECOMMENDED_COUNT == 3
    assert agent.SYSTEM_PROMPT is not None
    assert isinstance(agent.SYSTEM_PROMPT, str)
    assert len(agent.SYSTEM_PROMPT) > 0


def test_appetizer_agent_initialization_with_provider():
    """Test that AppetizerAgent can be initialized with provider."""
    agent = AppetizerAgent(provider="openai")
    assert agent.name == "appetizer_agent"
    assert agent is not None


def test_main_dish_agent_is_correctly_initialized():
    """Test that MainDishAgent is correctly initialized."""
    agent = MainDishAgent()
    assert agent.name == "main_dish_agent"
    assert agent.CATEGORY == "main_dish"
    assert agent.COURSE_NAME == "main_dish_expert"
    assert agent.RECOMMENDED_COUNT == 2
    assert agent.SYSTEM_PROMPT is not None
    assert isinstance(agent.SYSTEM_PROMPT, str)
    assert (
        "main dish" in agent.SYSTEM_PROMPT.lower()
        or "primo" in agent.SYSTEM_PROMPT.lower()
    )


def test_second_plate_agent_is_correctly_initialized():
    """Test that SecondPlateAgent is correctly initialized."""
    agent = SecondPlateAgent()
    assert agent.name == "second_plate_agent"
    assert agent.CATEGORY == "second_plate"
    assert agent.COURSE_NAME == "second_plate_expert"
    assert agent.RECOMMENDED_COUNT == 2
    assert agent.SYSTEM_PROMPT is not None
    assert isinstance(agent.SYSTEM_PROMPT, str)
    assert (
        "second" in agent.SYSTEM_PROMPT.lower()
        or "secondo" in agent.SYSTEM_PROMPT.lower()
    )


def test_dessert_agent_is_correctly_initialized():
    """Test that DessertAgent is correctly initialized."""
    agent = DessertAgent()
    assert agent.name == "dessert_agent"
    assert agent.CATEGORY == "dessert"
    assert agent.COURSE_NAME == "dessert_expert"
    assert agent.RECOMMENDED_COUNT == 2
    assert agent.SYSTEM_PROMPT is not None
    assert isinstance(agent.SYSTEM_PROMPT, str)
    assert "dessert" in agent.SYSTEM_PROMPT.lower()


def test_recipe_research_agent_is_correctly_initialized():
    """Test that RecipeResearchAgent is correctly initialized."""
    agent = RecipeResearchAgent()
    assert agent.name == "recipe_researcher"
    assert agent.system_prompt is not None
    assert isinstance(agent.system_prompt, str)
    assert len(agent.system_prompt) > 0


def test_base_recipe_agent_extract_preferences():
    """Test that _extract_preferences correctly extracts preferences."""
    agent = AppetizerAgent()
    preferences = {
        "has_vegans": True,
        "has_vegetarians": False,
        "prefer_traditional": True,
        "allergies": ["gluten"],
        "custom_allergies": ["peanuts"],
        "number_of_guests": 6,
        "vegan_count": 2,
        "vegetarian_count": 0,
    }

    result = agent._extract_preferences(preferences)

    assert isinstance(result, dict)
    assert "has_vegans" in result
    assert "has_vegetarians" in result
    assert "prefer_traditional" in result
    assert "allergens" in result
    assert "number_of_guests" in result
    assert "vegan_count" in result
    assert "vegetarian_count" in result
    assert isinstance(result["has_vegans"], bool)
    assert isinstance(result["has_vegetarians"], bool)
    assert isinstance(result["prefer_traditional"], bool)
    assert isinstance(result["allergens"], list)
    assert isinstance(result["vegan_count"], int)
    assert isinstance(result["vegetarian_count"], int)
    assert result["has_vegans"] is True
    assert result["has_vegetarians"] is False
    assert result["prefer_traditional"] is True
    assert "gluten" in result["allergens"]
    assert "peanuts" in result["allergens"]
    assert len(result["allergens"]) == 2
    assert result["number_of_guests"] == 6
    assert result["vegan_count"] == 2
    assert result["vegetarian_count"] == 0


def test_base_recipe_agent_extract_preferences_defaults():
    """Test that _extract_preferences uses defaults correctly."""
    agent = MainDishAgent()
    preferences = {}

    result = agent._extract_preferences(preferences)

    assert isinstance(result, dict)
    assert result["has_vegans"] is False
    assert result["has_vegetarians"] is False
    assert result["prefer_traditional"] is True
    assert isinstance(result["allergens"], list)
    assert result["allergens"] == []
    assert result["number_of_guests"] == "unknown"
    assert isinstance(result["number_of_guests"], str)
    assert result["vegan_count"] == 0
    assert result["vegetarian_count"] == 0


def test_base_recipe_agent_extract_preferences_combines_allergies():
    """Test that _extract_preferences correctly combines allergies and custom_allergies."""
    agent = DessertAgent()
    preferences = {
        "allergies": ["gluten", "nuts"],
        "custom_allergies": ["peanuts", "soy"],
    }

    result = agent._extract_preferences(preferences)

    assert isinstance(result["allergens"], list)
    assert len(result["allergens"]) == 4
    assert "gluten" in result["allergens"]
    assert "nuts" in result["allergens"]
    assert "peanuts" in result["allergens"]
    assert "soy" in result["allergens"]


def test_base_recipe_agent_extract_preferences_empty_allergies():
    """Test that _extract_preferences handles empty allergy lists."""
    agent = SecondPlateAgent()
    preferences = {
        "allergies": [],
        "custom_allergies": [],
    }

    result = agent._extract_preferences(preferences)

    assert isinstance(result["allergens"], list)
    assert len(result["allergens"]) == 0


def test_base_recipe_agent_build_prompt():
    """Test that _build_prompt creates a proper prompt."""
    agent = DessertAgent()
    prefs = {
        "has_vegans": True,
        "has_vegetarians": False,
        "prefer_traditional": True,
        "allergens": ["nuts"],
        "number_of_guests": 8,
        "vegan_count": 2,
        "vegetarian_count": 0,
    }

    prompt = agent._build_prompt(prefs, context="Test context")

    assert isinstance(prompt, str)
    assert len(prompt) > 0
    assert "8" in prompt
    assert "2" in prompt
    assert "nuts" in prompt
    assert "Test context" in prompt
    assert agent.COURSE_NAME in prompt.lower() or "dessert" in prompt.lower()
    assert str(agent.RECOMMENDED_COUNT) in prompt
    assert "vegan" in prompt.lower() or "vegetarian" in prompt.lower()


def test_base_recipe_agent_build_prompt_without_context():
    """Test that _build_prompt works without context."""
    agent = SecondPlateAgent()
    prefs = {
        "has_vegans": False,
        "has_vegetarians": False,
        "prefer_traditional": False,
        "allergens": [],
        "number_of_guests": 4,
        "vegan_count": 0,
        "vegetarian_count": 0,
    }

    prompt = agent._build_prompt(prefs)

    assert isinstance(prompt, str)
    assert len(prompt) > 0
    assert "4" in prompt
    assert "None" in prompt or "0" in prompt
    assert agent.CATEGORY in prompt.lower() or agent.COURSE_NAME in prompt.lower()


def test_base_recipe_agent_build_prompt_with_traditional_options():
    """Test that _build_prompt includes traditional options when available."""
    agent = AppetizerAgent()
    prefs = {
        "has_vegans": False,
        "has_vegetarians": False,
        "prefer_traditional": True,
        "allergens": [],
        "number_of_guests": 6,
        "vegan_count": 0,
        "vegetarian_count": 0,
    }

    prompt = agent._build_prompt(prefs)

    assert isinstance(prompt, str)
    assert "traditional" in prompt.lower()


@patch.object(RecipeResearchAgent, "run")
def test_recipe_research_agent_search_web_recipes(mock_run):
    """Test that search_web_recipes method exists and returns correct type."""
    mock_response = Mock()
    mock_response.text = "Search results for recipes"
    mock_run.return_value = mock_response

    agent = RecipeResearchAgent()

    assert hasattr(agent, "search_web_recipes")
    assert callable(agent.search_web_recipes)

    result = agent.search_web_recipes("Christmas cookies", "vegan")

    assert isinstance(result, str)
    assert len(result) > 0
    assert result == "Search results for recipes"
    mock_run.assert_called_once()

    call_args = mock_run.call_args
    assert call_args is not None
    prompt = call_args[0][0]
    assert isinstance(prompt, str)
    assert "Christmas cookies" in prompt
    assert "vegan" in prompt


@patch.object(RecipeResearchAgent, "run")
def test_recipe_research_agent_search_web_recipes_without_dietary(mock_run):
    """Test that search_web_recipes works without dietary requirements."""
    mock_response = Mock()
    mock_response.text = "Search results"
    mock_run.return_value = mock_response

    agent = RecipeResearchAgent()

    result = agent.search_web_recipes("Christmas dinner")

    assert isinstance(result, str)
    mock_run.assert_called_once()

    call_args = mock_run.call_args
    prompt = call_args[0][0]
    assert "Christmas dinner" in prompt


def test_base_recipe_agent_get_tools():
    """Test that _get_tools returns the correct tools."""
    agent = AppetizerAgent()
    tools = agent._get_tools()

    assert isinstance(tools, list)
    assert len(tools) > 0
    assert tools == agent.TOOLS


def test_base_recipe_agent_get_tools_different_agents():
    """Test that different agents have different tools."""
    appetizer_agent = AppetizerAgent()
    main_dish_agent = MainDishAgent()
    dessert_agent = DessertAgent()

    appetizer_tools = appetizer_agent._get_tools()
    main_dish_tools = main_dish_agent._get_tools()
    dessert_tools = dessert_agent._get_tools()

    assert isinstance(appetizer_tools, list)
    assert isinstance(main_dish_tools, list)
    assert isinstance(dessert_tools, list)
    assert len(appetizer_tools) > 0
    assert len(main_dish_tools) > 0
    assert len(dessert_tools) > 0


def test_base_recipe_agent_search_method_exists():
    """Test that search method exists on recipe agents."""
    agent = AppetizerAgent()

    assert hasattr(agent, "search")
    assert callable(agent.search)


@patch.object(AppetizerAgent, "run")
def test_base_recipe_agent_search_return_type_and_structure(mock_run):
    """Test that search() returns correct type and structure."""
    mock_response = Mock()
    mock_response.text = "Recipe suggestions"
    mock_run.return_value = mock_response

    agent = AppetizerAgent()
    preferences = {
        "number_of_guests": 6,
        "has_vegans": False,
        "has_vegetarians": True,
        "vegetarian_count": 2,
        "allergies": ["gluten"],
    }

    result = agent.search(preferences)

    assert isinstance(result, dict)
    assert "category" in result
    assert "raw_response" in result
    assert "preferences_used" in result

    assert isinstance(result["category"], str)
    assert isinstance(result["raw_response"], str)
    assert isinstance(result["preferences_used"], dict)

    assert result["category"] == agent.CATEGORY
    assert result["raw_response"] == "Recipe suggestions"
    assert result["preferences_used"]["number_of_guests"] == 6
    assert result["preferences_used"]["has_vegetarians"] is True

    mock_run.assert_called_once()
    call_args = mock_run.call_args
    assert call_args[1]["tool_choice"] == "required_first"


@patch.object(MainDishAgent, "run")
def test_base_recipe_agent_search_with_context(mock_run):
    """Test that search() accepts and uses context parameter."""
    mock_response = Mock()
    mock_response.text = "Main dish suggestions"
    mock_run.return_value = mock_response

    agent = MainDishAgent()
    preferences = {"number_of_guests": 4}

    result = agent.search(preferences, context="Test context")

    assert isinstance(result, dict)
    assert result["category"] == agent.CATEGORY

    call_args = mock_run.call_args
    prompt = call_args[0][0]
    assert "Test context" in prompt


@patch.object(DessertAgent, "run")
def test_base_recipe_agent_search_calls_extract_and_build(mock_run):
    """Test that search() calls _extract_preferences and _build_prompt."""
    mock_response = Mock()
    mock_response.text = "Dessert suggestions"
    mock_run.return_value = mock_response

    agent = DessertAgent()
    preferences = {
        "number_of_guests": 8,
        "has_vegans": True,
        "vegan_count": 2,
        "allergies": ["nuts"],
    }

    result = agent.search(preferences)

    assert isinstance(result, dict)
    assert result["preferences_used"]["number_of_guests"] == 8
    assert result["preferences_used"]["has_vegans"] is True
    assert "nuts" in result["preferences_used"]["allergens"]

    mock_run.assert_called_once()
    call_args = mock_run.call_args
    prompt = call_args[0][0]
    assert "8" in prompt
    assert "2" in prompt
    assert "nuts" in prompt


def test_base_recipe_agent_search_all_agents():
    """Test that search() method exists on all recipe agents."""
    agents = [
        AppetizerAgent(),
        MainDishAgent(),
        SecondPlateAgent(),
        DessertAgent(),
    ]

    for agent in agents:
        assert hasattr(agent, "search")
        assert callable(agent.search)
        assert agent.CATEGORY is not None
        assert agent.COURSE_NAME is not None
        assert isinstance(agent.RECOMMENDED_COUNT, int)
        assert agent.RECOMMENDED_COUNT > 0
