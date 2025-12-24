from unittest.mock import Mock, patch

import pytest

from src.agents.menu_creator import MenuCreatorAgent, format_final_menu


def test_menu_creator_is_correctly_initialized():
    """Test that MenuCreatorAgent is correctly initialized."""
    agent = MenuCreatorAgent()
    assert agent.name == "menu_creator"
    assert agent.system_prompt is not None
    assert agent._max_steps == 10
    assert agent._planning_interval == 3
    assert agent._planning_prompt is not None
    assert agent._terminate_on_text is True


def test_menu_creator_initialization_with_recipe_agents():
    """Test that MenuCreatorAgent can be initialized with recipe agents."""
    from src.agents.recipe_agents import AppetizerAgent, MainDishAgent

    recipe_agents = [AppetizerAgent(), MainDishAgent()]
    agent = MenuCreatorAgent(recipe_agents=recipe_agents)

    assert agent.name == "menu_creator"
    assert agent is not None


def test_menu_creator_extract_preferences_missing_guests():
    """Test that _extract_preferences raises error when guests are missing."""
    agent = MenuCreatorAgent()
    preferences = {
        "has_vegans": True,
    }

    with pytest.raises(ValueError, match="Number of guests is mandatory"):
        agent._extract_preferences(preferences)


def test_menu_creator_extract_preferences_complete():
    """Test that _extract_preferences correctly extracts all preferences."""
    agent = MenuCreatorAgent()
    preferences = {
        "number_of_guests": 8,
        "has_vegans": True,
        "has_vegetarians": False,
        "allergies": ["gluten", "nuts"],
        "custom_allergies": ["peanuts"],
    }

    result = agent._extract_preferences(preferences)

    assert isinstance(result, dict)

    assert "number_of_guests" in result
    assert "has_vegans" in result
    assert "has_vegetarians" in result
    assert "allergens" in result

    assert result["number_of_guests"] == 8
    assert result["has_vegans"] is True
    assert result["has_vegetarians"] is False
    assert isinstance(result["allergens"], list)
    assert len(result["allergens"]) == 3
    assert "gluten" in result["allergens"]
    assert "nuts" in result["allergens"]
    assert "peanuts" in result["allergens"]


def test_menu_creator_extract_preferences_with_defaults():
    """Test that _extract_preferences uses defaults correctly."""
    agent = MenuCreatorAgent()
    preferences = {
        "number_of_guests": 4,
    }

    result = agent._extract_preferences(preferences)

    assert result["number_of_guests"] == 4
    assert result["has_vegans"] is False
    assert result["has_vegetarians"] is False
    assert isinstance(result["allergens"], list)
    assert len(result["allergens"]) == 0


def test_menu_creator_build_menu_prompt():
    """Test that _build_menu_prompt creates a proper prompt."""
    agent = MenuCreatorAgent()
    preferences = {
        "number_of_guests": 6,
        "has_vegans": False,
        "has_vegetarians": True,
        "vegetarian_count": 2,
        "allergies": ["nuts"],
        "prefer_traditional": True,
    }

    prompt = agent._build_menu_prompt(
        preferences,
        appetizer_suggestions="Test appetizers",
        main_dish_suggestions="Test main dishes",
        second_plate_suggestions="Test second plates",
        dessert_suggestions="Test desserts",
    )

    assert isinstance(prompt, str)
    assert len(prompt) > 0
    assert "6" in prompt
    assert "Test appetizers" in prompt
    assert "Test main dishes" in prompt
    assert "Test second plates" in prompt
    assert "Test desserts" in prompt
    assert "nuts" in prompt
    assert "GUEST INFORMATION" in prompt
    assert "APPETIZER SUGGESTIONS" in prompt
    assert "MAIN DISH SUGGESTIONS" in prompt
    assert "SECOND PLATE SUGGESTIONS" in prompt
    assert "DESSERT SUGGESTIONS" in prompt


def test_menu_creator_build_menu_prompt_with_traditional_flag():
    """Test that _build_menu_prompt respects traditional parameter."""
    agent = MenuCreatorAgent()
    preferences = {
        "number_of_guests": 4,
        "prefer_traditional": False,
    }

    prompt = agent._build_menu_prompt(
        preferences,
        traditional=True,
    )

    assert "traditional" in prompt.lower() or "Yes" in prompt


def test_menu_creator_build_menu_prompt_empty_suggestions():
    """Test that _build_menu_prompt handles empty suggestions."""
    agent = MenuCreatorAgent()
    preferences = {
        "number_of_guests": 2,
    }

    prompt = agent._build_menu_prompt(
        preferences,
        appetizer_suggestions="",
        main_dish_suggestions="",
        second_plate_suggestions="",
        dessert_suggestions="",
    )

    assert isinstance(prompt, str)
    assert "2" in prompt


def test_format_final_menu_tool():
    """Test that format_final_menu tool formats menu correctly."""
    menu = format_final_menu(
        title="Christmas Dinner 2024",
        appetizers="Test appetizer 1\nTest appetizer 2",
        main_dishes="Test main dish",
        second_plates="Test second plate",
        desserts="Test dessert",
        number_of_guests=8,
        preparation_notes="Start early",
        shopping_tips="Buy fresh ingredients",
    )

    assert isinstance(menu, str)
    assert len(menu) > 0
    assert "Christmas Dinner 2024" in menu
    assert "8" in menu
    assert "Test appetizer 1" in menu
    assert "Test main dish" in menu
    assert "Test second plate" in menu
    assert "Test dessert" in menu
    assert "Start early" in menu
    assert "Buy fresh ingredients" in menu
    assert "APPETIZERS" in menu
    assert "MAIN DISHES" in menu
    assert "SECOND PLATES" in menu
    assert "DESSERTS" in menu
    assert "PREPARATION TIMELINE" in menu
    assert "SHOPPING TIPS" in menu
    assert "Happy Christmas" in menu


def test_format_final_menu_tool_type_signature():
    """Test that format_final_menu accepts correct types."""
    menu = format_final_menu(
        title="Test",
        appetizers="App",
        main_dishes="Main",
        second_plates="Second",
        desserts="Dessert",
        number_of_guests=5,
    )

    assert isinstance(menu, str)

    menu_with_optional = format_final_menu(
        title="Test",
        appetizers="App",
        main_dishes="Main",
        second_plates="Second",
        desserts="Dessert",
        number_of_guests=5,
        preparation_notes="Notes",
        shopping_tips="Tips",
    )

    assert isinstance(menu_with_optional, str)
    assert "Notes" in menu_with_optional
    assert "Tips" in menu_with_optional


def test_format_final_menu_without_optional_fields():
    """Test format_final_menu without optional fields."""
    menu = format_final_menu(
        title="Test Menu",
        appetizers="Appetizer",
        main_dishes="Main",
        second_plates="Second",
        desserts="Dessert",
        number_of_guests=4,
    )

    assert "Test Menu" in menu
    assert "4" in menu
    assert "PREPARATION TIMELINE" not in menu
    assert "SHOPPING TIPS" not in menu


def test_menu_creator_connect_recipe_agents():
    """Test that connect_recipe_agents connects agents correctly."""
    from src.agents.recipe_agents import AppetizerAgent, MainDishAgent

    agent = MenuCreatorAgent()
    recipe_agents = [
        AppetizerAgent(),
        MainDishAgent(),
    ]

    assert hasattr(agent, "connect_recipe_agents")
    assert callable(agent.connect_recipe_agents)
    agent.connect_recipe_agents(recipe_agents)
    assert agent is not None


def test_menu_creator_connect_recipe_agents_empty_list():
    """Test that connect_recipe_agents handles empty list."""
    agent = MenuCreatorAgent()

    agent.connect_recipe_agents([])
    assert agent is not None


@patch.object(MenuCreatorAgent, "run")
def test_create_menu_return_type_and_structure(mock_run):
    """Test that create_menu returns correct type and structure."""
    mock_response = Mock()
    mock_response.text = "Mock menu response"
    mock_run.return_value = mock_response

    agent = MenuCreatorAgent()
    preferences = {
        "number_of_guests": 6,
        "has_vegans": True,
        "has_vegetarians": False,
        "allergies": ["gluten"],
    }

    result = agent.create_menu(
        preferences=preferences,
        appetizer_suggestions="Appetizer suggestions",
        main_dish_suggestions="Main dish suggestions",
        second_plate_suggestions="Second plate suggestions",
        dessert_suggestions="Dessert suggestions",
    )

    assert isinstance(result, dict)
    assert "formatted_menu" in result
    assert "number_of_guests" in result
    assert "dietary_accommodations" in result
    assert isinstance(result["formatted_menu"], str)
    assert isinstance(result["number_of_guests"], int)
    assert isinstance(result["dietary_accommodations"], dict)
    dietary = result["dietary_accommodations"]
    assert "vegan_options" in dietary
    assert "vegetarian_options" in dietary
    assert "allergens_avoided" in dietary
    assert isinstance(dietary["vegan_options"], bool)
    assert isinstance(dietary["vegetarian_options"], bool)
    assert isinstance(dietary["allergens_avoided"], list)
    assert result["number_of_guests"] == 6
    assert dietary["vegan_options"] is True
    assert dietary["vegetarian_options"] is False
    assert "gluten" in dietary["allergens_avoided"]
    mock_run.assert_called_once()


@patch.object(MenuCreatorAgent, "run")
def test_create_menu_calls_build_prompt(mock_run):
    """Test that create_menu calls _build_menu_prompt correctly."""
    mock_response = Mock()
    mock_response.text = "Menu"
    mock_run.return_value = mock_response

    agent = MenuCreatorAgent()
    preferences = {
        "number_of_guests": 4,
    }

    agent.create_menu(
        preferences=preferences,
        appetizer_suggestions="A",
        main_dish_suggestions="M",
        second_plate_suggestions="S",
        dessert_suggestions="D",
    )

    call_args = mock_run.call_args
    assert call_args is not None
    prompt = call_args[0][0]
    assert isinstance(prompt, str)
    assert "A" in prompt or "M" in prompt or "S" in prompt or "D" in prompt


@patch.object(MenuCreatorAgent, "run")
def test_create_menu_with_agents_return_structure(mock_run):
    """Test that create_menu_with_agents returns correct structure."""
    mock_response = Mock()
    mock_response.text = "Menu from agents"
    mock_run.return_value = mock_response

    agent = MenuCreatorAgent()
    preferences = {
        "number_of_guests": 8,
        "has_vegans": False,
        "has_vegetarians": True,
        "allergies": [],
    }

    result = agent.create_menu_with_agents(preferences)

    assert isinstance(result, dict)
    assert "formatted_menu" in result
    assert "number_of_guests" in result
    assert "dietary_accommodations" in result
    assert result["number_of_guests"] == 8
    assert result["dietary_accommodations"]["vegetarian_options"] is True
    assert result["dietary_accommodations"]["vegan_options"] is False
