import pytest
from src.config import settings, get_model_name, get_provider_name
from src.agents.info_checker import InfoCheckerAgent

def test_settings_should_return_correct_infos(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    provider = get_provider_name()
    model = get_model_name()

    assert model == 'gpt-4o-mini'
    assert provider == 'OpenAI'
    assert settings.LLM_PROVIDER == 'openai'
    assert settings.is_ollama() is False

def test_agent_info_checker_is_correctly_initialized():
    agent = InfoCheckerAgent()
    assert agent.name == 'info_checker'

def test_extract_preferences_missing_guests():
    agent = InfoCheckerAgent()
    result = agent.extract_preferences("I want a Christmas menu")

    assert result["is_complete"] is False
    assert "number_of_guests" in result["missing_info"]
    assert len(result["questions"])>0
    assert isinstance(result["raw_response"], str)

def test_agent_info_checker_should_correctly_extract_preferences():
    agent = InfoCheckerAgent()
    request = 'I need a Christmas dinner menu for 8 people. Two guests are vegetarian and one has a nut allergy'
    result = agent.extract_preferences(request)
    
    
