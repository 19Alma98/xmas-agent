from unittest.mock import Mock, patch, MagicMock
from src.agents.orchestrator import ChristmasMenuOrchestrator, create_orchestrator


@patch("src.agents.orchestrator.RecipeVectorStore")
def test_orchestrator_is_correctly_initialized(mock_vector_store_class):
    """Test that ChristmasMenuOrchestrator is correctly initialized."""
    mock_store = MagicMock()
    mock_store.count_recipes.return_value = 0
    mock_vector_store_class.return_value = mock_store

    orchestrator = ChristmasMenuOrchestrator(initialize_db=False)

    assert orchestrator.info_checker is not None
    assert orchestrator.appetizer_agent is not None
    assert orchestrator.main_dish_agent is not None
    assert orchestrator.second_plate_agent is not None
    assert orchestrator.dessert_agent is not None
    assert orchestrator.recipe_researcher is not None
    assert orchestrator.menu_creator is not None
    assert orchestrator.vector_store is not None

    assert hasattr(orchestrator, "api_key")
    assert hasattr(orchestrator, "provider")
    assert isinstance(orchestrator.provider, str)


@patch("src.agents.orchestrator.RecipeVectorStore")
def test_orchestrator_initialization_with_provider(mock_vector_store_class):
    """Test that orchestrator can be initialized with a specific provider."""
    mock_store = MagicMock()
    mock_store.count_recipes.return_value = 0
    mock_vector_store_class.return_value = mock_store

    orchestrator = ChristmasMenuOrchestrator(provider="openai", initialize_db=False)

    assert orchestrator.provider == "openai"
    assert orchestrator.info_checker is not None
    assert orchestrator.menu_creator is not None


@patch("src.agents.orchestrator.RecipeVectorStore")
def test_orchestrator_initialization_with_api_key(mock_vector_store_class):
    """Test that orchestrator can be initialized with an API key."""
    mock_store = MagicMock()
    mock_store.count_recipes.return_value = 0
    mock_vector_store_class.return_value = mock_store

    orchestrator = ChristmasMenuOrchestrator(api_key="test-key", initialize_db=False)

    assert orchestrator.api_key == "test-key"
    assert orchestrator.info_checker is not None


@patch("src.agents.orchestrator.RecipeVectorStore")
def test_orchestrator_get_provider_info(mock_vector_store_class):
    """Test that get_provider_info returns correct information."""
    mock_store = MagicMock()
    mock_store.count_recipes.return_value = 0
    mock_vector_store_class.return_value = mock_store

    orchestrator = ChristmasMenuOrchestrator(initialize_db=False)
    provider_info = orchestrator.get_provider_info()

    assert isinstance(provider_info, str)
    assert len(provider_info) > 0
    assert "(" in provider_info
    assert ")" in provider_info


@patch("src.agents.orchestrator.RecipeVectorStore")
def test_orchestrator_get_provider_info_format(mock_vector_store_class):
    """Test that get_provider_info returns formatted string."""
    mock_store = MagicMock()
    mock_store.count_recipes.return_value = 0
    mock_vector_store_class.return_value = mock_store

    orchestrator = ChristmasMenuOrchestrator(initialize_db=False)
    provider_info = orchestrator.get_provider_info()

    assert isinstance(provider_info, str)
    parts = provider_info.split("(")
    assert len(parts) == 2
    assert len(parts[0].strip()) > 0


@patch("src.agents.orchestrator.RecipeVectorStore")
def test_orchestrator_get_recipe_count(mock_vector_store_class):
    """Test that get_recipe_count returns a count."""
    mock_store = MagicMock()
    mock_store.count_recipes.return_value = 5
    mock_vector_store_class.return_value = mock_store

    orchestrator = ChristmasMenuOrchestrator(initialize_db=False)
    count = orchestrator.get_recipe_count()

    assert isinstance(count, int)
    assert count == 5
    mock_store.count_recipes.assert_called()


@patch("src.agents.orchestrator.RecipeVectorStore")
def test_orchestrator_search_web_recipes(mock_vector_store_class):
    """Test that search_web_recipes method exists and can be called."""
    mock_store = MagicMock()
    mock_store.count_recipes.return_value = 0
    mock_vector_store_class.return_value = mock_store

    orchestrator = ChristmasMenuOrchestrator(initialize_db=False)

    assert hasattr(orchestrator, "search_web_recipes")
    assert callable(orchestrator.search_web_recipes)

    with patch.object(
        orchestrator.recipe_researcher,
        "search_web_recipes",
        return_value="Test results",
    ) as mock_search:
        result = orchestrator.search_web_recipes("test query", "vegan")

        assert isinstance(result, str)
        assert result == "Test results"
        mock_search.assert_called_once_with("test query", "vegan")


@patch("src.agents.orchestrator.RecipeVectorStore")
def test_orchestrator_search_web_recipes_without_dietary(mock_vector_store_class):
    """Test that search_web_recipes works without dietary requirements."""
    mock_store = MagicMock()
    mock_store.count_recipes.return_value = 0
    mock_vector_store_class.return_value = mock_store

    orchestrator = ChristmasMenuOrchestrator(initialize_db=False)

    with patch.object(
        orchestrator.recipe_researcher, "search_web_recipes", return_value="Results"
    ) as mock_search:
        result = orchestrator.search_web_recipes("test query")

        assert isinstance(result, str)
        mock_search.assert_called_once_with("test query", "")


@patch("src.agents.orchestrator.RecipeVectorStore")
def test_orchestrator_clear_info_checker_memory(mock_vector_store_class):
    """Test that clear_info_checker_memory clears memory."""
    mock_store = MagicMock()
    mock_store.count_recipes.return_value = 0
    mock_vector_store_class.return_value = mock_store

    orchestrator = ChristmasMenuOrchestrator(initialize_db=False)

    assert hasattr(orchestrator, "clear_info_checker_memory")
    assert callable(orchestrator.clear_info_checker_memory)

    with patch.object(orchestrator.info_checker, "clear_memory") as mock_clear:
        orchestrator.clear_info_checker_memory()
        mock_clear.assert_called_once()

    assert orchestrator.info_checker is not None


@patch("src.agents.orchestrator.RecipeVectorStore")
def test_create_orchestrator_factory(mock_vector_store_class):
    """Test that create_orchestrator factory function works."""
    mock_store = MagicMock()
    mock_store.count_recipes.return_value = 0
    mock_vector_store_class.return_value = mock_store

    orchestrator = create_orchestrator()

    assert isinstance(orchestrator, ChristmasMenuOrchestrator)
    assert orchestrator.info_checker is not None
    assert orchestrator.menu_creator is not None


@patch("src.agents.orchestrator.RecipeVectorStore")
def test_create_orchestrator_with_provider(mock_vector_store_class):
    """Test that create_orchestrator accepts provider parameter."""
    mock_store = MagicMock()
    mock_store.count_recipes.return_value = 0
    mock_vector_store_class.return_value = mock_store

    orchestrator = create_orchestrator(provider="openai")

    assert isinstance(orchestrator, ChristmasMenuOrchestrator)
    assert orchestrator.provider == "openai"


@patch("src.agents.orchestrator.RecipeVectorStore")
def test_create_orchestrator_with_api_key(mock_vector_store_class):
    """Test that create_orchestrator accepts api_key parameter."""
    mock_store = MagicMock()
    mock_store.count_recipes.return_value = 0
    mock_vector_store_class.return_value = mock_store

    orchestrator = create_orchestrator(api_key="test-key-123")

    assert isinstance(orchestrator, ChristmasMenuOrchestrator)
    assert orchestrator.api_key == "test-key-123"


@patch("src.agents.orchestrator.RecipeVectorStore")
def test_create_orchestrator_with_both_params(mock_vector_store_class):
    """Test that create_orchestrator accepts both api_key and provider."""
    mock_store = MagicMock()
    mock_store.count_recipes.return_value = 0
    mock_vector_store_class.return_value = mock_store

    orchestrator = create_orchestrator(api_key="test-key", provider="openai")

    assert isinstance(orchestrator, ChristmasMenuOrchestrator)
    assert orchestrator.api_key == "test-key"
    assert orchestrator.provider == "openai"


@patch("src.agents.orchestrator.RecipeLoader")
@patch("src.agents.orchestrator.RecipeVectorStore")
def test_orchestrator_reload_recipes(mock_vector_store_class, mock_loader_class):
    """Test that reload_recipes method exists and can be called."""
    mock_store = MagicMock()
    mock_store.count_recipes.return_value = 0
    mock_store.clear_all.return_value = None
    mock_vector_store_class.return_value = mock_store

    mock_loader = MagicMock()
    mock_loader.load_sample_recipes.return_value = 10
    mock_loader_class.return_value = mock_loader

    orchestrator = ChristmasMenuOrchestrator(initialize_db=False)

    assert hasattr(orchestrator, "reload_recipes")
    assert callable(orchestrator.reload_recipes)

    count = orchestrator.reload_recipes()
    assert isinstance(count, int)
    assert count == 10
    mock_store.clear_all.assert_called()
    mock_loader.load_sample_recipes.assert_called()


@patch("src.agents.orchestrator.RecipeLoader")
@patch("src.agents.orchestrator.RecipeVectorStore")
def test_orchestrator_reload_recipes_with_json_file(
    mock_vector_store_class, mock_loader_class
):
    """Test that reload_recipes accepts json_file parameter."""
    mock_store = MagicMock()
    mock_store.count_recipes.return_value = 0
    mock_store.clear_all.return_value = None
    mock_vector_store_class.return_value = mock_store

    mock_loader = MagicMock()
    mock_loader.load_sample_recipes.return_value = 5
    mock_loader_class.return_value = mock_loader

    orchestrator = ChristmasMenuOrchestrator(initialize_db=False)

    count = orchestrator.reload_recipes(json_file=None)
    assert isinstance(count, int)
    assert count == 5
    mock_loader.load_sample_recipes.assert_called()


@patch("src.agents.orchestrator.RecipeVectorStore")
@patch.object(ChristmasMenuOrchestrator, "_ensure_recipes_loaded")
def test_run_return_type_and_structure(mock_ensure, mock_vector_store_class):
    """Test that run() returns correct type and structure."""
    mock_store = MagicMock()
    mock_store.count_recipes.return_value = 0
    mock_vector_store_class.return_value = mock_store

    orchestrator = ChristmasMenuOrchestrator(initialize_db=False)

    mock_preferences_result = {
        "is_complete": True,
        "preferences": {
            "number_of_guests": 6,
            "has_vegans": False,
            "has_vegetarians": False,
            "allergies": [],
        },
        "raw_response": "Preferences extracted",
    }

    mock_recipe_result = {
        "category": "appetizer",
        "raw_response": "Recipe suggestions",
        "preferences_used": {},
    }

    mock_menu_result = {
        "formatted_menu": "Formatted menu",
        "number_of_guests": 6,
        "dietary_accommodations": {
            "vegan_options": False,
            "vegetarian_options": False,
            "allergens_avoided": [],
        },
    }

    with (
        patch.object(
            orchestrator.info_checker,
            "extract_preferences",
            return_value=mock_preferences_result,
        ),
        patch.object(
            orchestrator.appetizer_agent, "search", return_value=mock_recipe_result
        ),
        patch.object(
            orchestrator.main_dish_agent, "search", return_value=mock_recipe_result
        ),
        patch.object(
            orchestrator.second_plate_agent, "search", return_value=mock_recipe_result
        ),
        patch.object(
            orchestrator.dessert_agent, "search", return_value=mock_recipe_result
        ),
        patch.object(
            orchestrator.menu_creator, "create_menu", return_value=mock_menu_result
        ),
    ):
        result = orchestrator.run("I need a Christmas menu for 6 people")

        assert isinstance(result, dict)

        assert "success" in result
        assert "menu" in result
        assert "preferences" in result
        assert "agent_outputs" in result
        assert "error" in result
        assert "provider" in result

        assert isinstance(result["success"], bool)
        assert isinstance(result["menu"], str) or result["menu"] is None
        assert isinstance(result["preferences"], dict) or result["preferences"] is None
        assert isinstance(result["agent_outputs"], dict)
        assert isinstance(result["error"], str) or result["error"] is None
        assert isinstance(result["provider"], str)

        assert result["success"] is True
        assert result["menu"] == "Formatted menu"
        assert result["preferences"]["number_of_guests"] == 6
        assert "info_checker" in result["agent_outputs"]
        assert "appetizer" in result["agent_outputs"]
        assert "main_dish" in result["agent_outputs"]
        assert "second_plate" in result["agent_outputs"]
        assert "dessert" in result["agent_outputs"]
        assert "menu_creator" in result["agent_outputs"]


@patch("src.agents.orchestrator.RecipeVectorStore")
@patch.object(ChristmasMenuOrchestrator, "_ensure_recipes_loaded")
def test_run_with_incomplete_preferences(mock_ensure, mock_vector_store_class):
    """Test that run() handles incomplete preferences correctly."""
    mock_store = MagicMock()
    mock_store.count_recipes.return_value = 0
    mock_vector_store_class.return_value = mock_store

    orchestrator = ChristmasMenuOrchestrator(initialize_db=False)

    mock_preferences_result = {
        "is_complete": False,
        "preferences": None,
        "missing_info": ["number_of_guests"],
        "questions": ["How many guests?"],
    }

    mock_recipe_result = {"raw_response": "Suggestions"}
    mock_menu_result = {"formatted_menu": "Menu"}

    with (
        patch.object(
            orchestrator.info_checker,
            "extract_preferences",
            return_value=mock_preferences_result,
        ),
        patch.object(
            orchestrator.appetizer_agent, "search", return_value=mock_recipe_result
        ),
        patch.object(
            orchestrator.main_dish_agent, "search", return_value=mock_recipe_result
        ),
        patch.object(
            orchestrator.second_plate_agent, "search", return_value=mock_recipe_result
        ),
        patch.object(
            orchestrator.dessert_agent, "search", return_value=mock_recipe_result
        ),
        patch.object(
            orchestrator.menu_creator, "create_menu", return_value=mock_menu_result
        ),
    ):
        result = orchestrator.run("I need a menu", interactive=False)

        assert isinstance(result, dict)
        assert result["success"] is True
        assert result["preferences"] is not None
        assert result["preferences"]["number_of_guests"] == 6


@patch("src.agents.orchestrator.RecipeVectorStore")
@patch.object(ChristmasMenuOrchestrator, "_ensure_recipes_loaded")
def test_run_calls_all_agents(mock_ensure, mock_vector_store_class):
    """Test that run() calls all required agents."""
    mock_store = MagicMock()
    mock_store.count_recipes.return_value = 0
    mock_vector_store_class.return_value = mock_store

    orchestrator = ChristmasMenuOrchestrator(initialize_db=False)

    mock_preferences_result = {
        "is_complete": True,
        "preferences": {"number_of_guests": 4},
    }

    mock_recipe_result = {"raw_response": "Suggestions"}
    mock_menu_result = {"formatted_menu": "Menu"}

    with (
        patch.object(
            orchestrator.info_checker,
            "extract_preferences",
            return_value=mock_preferences_result,
        ) as mock_info,
        patch.object(
            orchestrator.appetizer_agent, "search", return_value=mock_recipe_result
        ) as mock_app,
        patch.object(
            orchestrator.main_dish_agent, "search", return_value=mock_recipe_result
        ) as mock_main,
        patch.object(
            orchestrator.second_plate_agent, "search", return_value=mock_recipe_result
        ) as mock_second,
        patch.object(
            orchestrator.dessert_agent, "search", return_value=mock_recipe_result
        ) as mock_dessert,
        patch.object(
            orchestrator.menu_creator, "create_menu", return_value=mock_menu_result
        ) as mock_menu,
    ):
        orchestrator.run("Menu for 4 people")

        mock_info.assert_called_once()
        mock_app.assert_called_once()
        mock_main.assert_called_once()
        mock_second.assert_called_once()
        mock_dessert.assert_called_once()
        mock_menu.assert_called_once()


@patch("src.agents.orchestrator.RecipeVectorStore")
def test_run_streaming_return_type(mock_vector_store_class):
    """Test that run_streaming returns a Generator."""
    mock_store = MagicMock()
    mock_store.count_recipes.return_value = 0
    mock_vector_store_class.return_value = mock_store

    orchestrator = ChristmasMenuOrchestrator(initialize_db=False)

    assert hasattr(orchestrator, "run_streaming")
    assert callable(orchestrator.run_streaming)

    import inspect

    assert inspect.isgeneratorfunction(orchestrator.run_streaming)


@patch("src.agents.orchestrator.RecipeVectorStore")
@patch.object(ChristmasMenuOrchestrator, "_ensure_recipes_loaded")
def test_run_streaming_yields_correct_structure(mock_ensure, mock_vector_store_class):
    """Test that run_streaming yields dictionaries with correct structure."""
    mock_store = MagicMock()
    mock_store.count_recipes.return_value = 0
    mock_vector_store_class.return_value = mock_store

    orchestrator = ChristmasMenuOrchestrator(initialize_db=False)

    mock_preferences_result = {
        "is_complete": True,
        "preferences": {"number_of_guests": 4},
    }

    mock_recipe_result = {"raw_response": "Suggestions"}
    mock_step = Mock()
    mock_step.text = "Step text"
    mock_step.index = 0

    with (
        patch.object(
            orchestrator.info_checker,
            "extract_preferences",
            return_value=mock_preferences_result,
        ),
        patch.object(
            orchestrator.appetizer_agent, "search", return_value=mock_recipe_result
        ),
        patch.object(
            orchestrator.main_dish_agent, "search", return_value=mock_recipe_result
        ),
        patch.object(
            orchestrator.second_plate_agent, "search", return_value=mock_recipe_result
        ),
        patch.object(
            orchestrator.dessert_agent, "search", return_value=mock_recipe_result
        ),
        patch.object(
            orchestrator.menu_creator,
            "create_menu_streaming",
            return_value=iter([mock_step]),
        ),
    ):
        results = list(orchestrator.run_streaming("Menu for 4"))

        assert len(results) > 0
        for result in results:
            assert isinstance(result, dict)
            assert "type" in result
            assert isinstance(result["type"], str)
            assert result["type"] in [
                "status",
                "progress",
                "warning",
                "step",
                "complete",
            ]


@patch("src.agents.orchestrator.RecipeVectorStore")
def test_run_async_is_async_function(mock_vector_store_class):
    """Test that run_async is an async function."""
    mock_store = MagicMock()
    mock_store.count_recipes.return_value = 0
    mock_vector_store_class.return_value = mock_store

    orchestrator = ChristmasMenuOrchestrator(initialize_db=False)

    assert hasattr(orchestrator, "run_async")
    assert callable(orchestrator.run_async)

    import inspect

    assert inspect.iscoroutinefunction(orchestrator.run_async)


@patch("src.agents.orchestrator.RecipeVectorStore")
def test_run_async_streaming_is_async_generator(mock_vector_store_class):
    """Test that run_async_streaming is an async generator."""
    mock_store = MagicMock()
    mock_store.count_recipes.return_value = 0
    mock_vector_store_class.return_value = mock_store

    orchestrator = ChristmasMenuOrchestrator(initialize_db=False)

    assert hasattr(orchestrator, "run_async_streaming")
    assert callable(orchestrator.run_async_streaming)

    import inspect

    assert inspect.isasyncgenfunction(orchestrator.run_async_streaming)


@patch("src.agents.orchestrator.RecipeVectorStore")
@patch.object(ChristmasMenuOrchestrator, "_ensure_recipes_loaded")
def test_run_with_native_agents_return_structure(mock_ensure, mock_vector_store_class):
    """Test that run_with_native_agents returns correct structure."""
    mock_store = MagicMock()
    mock_store.count_recipes.return_value = 0
    mock_vector_store_class.return_value = mock_store

    orchestrator = ChristmasMenuOrchestrator(initialize_db=False)

    mock_preferences_result = {
        "is_complete": True,
        "preferences": {"number_of_guests": 6},
    }

    mock_menu_result = {
        "formatted_menu": "Native menu",
        "number_of_guests": 6,
        "dietary_accommodations": {},
    }

    with (
        patch.object(
            orchestrator.info_checker,
            "extract_preferences",
            return_value=mock_preferences_result,
        ),
        patch.object(
            orchestrator.menu_creator,
            "create_menu_with_agents",
            return_value=mock_menu_result,
        ),
    ):
        result = orchestrator.run_with_native_agents("Menu for 6")

        assert isinstance(result, dict)
        assert "success" in result
        assert "menu" in result
        assert "preferences" in result
        assert "provider" in result

        assert result["success"] is True
        assert isinstance(result["menu"], str)
        assert isinstance(result["preferences"], dict)
        assert isinstance(result["provider"], str)


@patch("src.agents.orchestrator.RecipeVectorStore")
@patch.object(ChristmasMenuOrchestrator, "_ensure_recipes_loaded")
def test_run_with_native_agents_calls_menu_creator(
    mock_ensure, mock_vector_store_class
):
    """Test that run_with_native_agents calls menu_creator.create_menu_with_agents."""
    mock_store = MagicMock()
    mock_store.count_recipes.return_value = 0
    mock_vector_store_class.return_value = mock_store

    orchestrator = ChristmasMenuOrchestrator(initialize_db=False)

    mock_preferences_result = {
        "is_complete": True,
        "preferences": {"number_of_guests": 4},
    }

    mock_menu_result = {"formatted_menu": "Menu"}

    with (
        patch.object(
            orchestrator.info_checker,
            "extract_preferences",
            return_value=mock_preferences_result,
        ),
        patch.object(
            orchestrator.menu_creator,
            "create_menu_with_agents",
            return_value=mock_menu_result,
        ) as mock_create,
    ):
        orchestrator.run_with_native_agents("Menu")

        mock_create.assert_called_once()
        call_args = mock_create.call_args[0][0]
        assert call_args["number_of_guests"] == 4
