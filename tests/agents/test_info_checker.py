from src.config import settings, get_model_name, get_provider_name
from src.agents.info_checker import InfoCheckerAgent


def test_settings_should_return_correct_infos(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    provider = get_provider_name()
    model = get_model_name()

    assert model == "gpt-4o-mini"
    assert provider == "OpenAI"
    assert settings.LLM_PROVIDER == "openai"
    assert settings.is_ollama() is False


def test_agent_info_checker_is_correctly_initialized():
    agent = InfoCheckerAgent()
    assert agent.name == "info_checker"


def test_extract_preferences_missing_guests():
    agent = InfoCheckerAgent()
    result = agent.extract_preferences("I want a Christmas menu")

    assert result["is_complete"] is False
    assert "number_of_guests" in result["missing_info"]
    assert len(result["questions"]) > 0
    assert isinstance(result["raw_response"], str)


def test_agent_info_checker_should_correctly_extract_preferences():
    agent = InfoCheckerAgent()
    request = "I need a Christmas dinner menu for 8 people. Two guests are vegetarian and one has a gluten allergy."
    result = agent.extract_preferences(request)

    assert isinstance(result["is_complete"], bool)
    assert result["preferences"]["number_of_guests"] == 8
    assert result["preferences"]["has_vegetarians"] is True
    assert result["preferences"]["vegetarian_count"] == 2
    assert result["preferences"]["allergies"] == ["gluten"]
    assert isinstance(result["raw_response"], str)
    assert isinstance(result["questions"], list)
    assert "8" in result["summary"] or "eight" in result["summary"].lower()
    assert "2" in result["summary"] or "two" in result["summary"].lower()
    assert "1" in result["summary"] or "one" in result["summary"].lower()
    assert "gluten" in result["summary"].lower()
