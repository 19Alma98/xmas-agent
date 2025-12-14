# 🎄 Christmas Menu Planner

A multi-agent AI system built with **Datapizza AI** that creates personalized Christmas dinner menus based on your requirements.

## 🌟 Features

### Core Features
- **Multi-Agent Architecture**: Specialized agents work together to create the perfect menu
- **RAG-Powered Recipe Search**: Semantic search with filters for dietary requirements
- **Dietary Accommodations**: Supports vegan, vegetarian, gluten-free, and allergy-aware menus
- **Traditional Italian Focus**: Includes classic Christmas recipes with modern alternatives
- **Interactive Mode**: Conversational interface for menu planning

### Datapizza AI Features Used

| Feature | Description | Implementation |
|---------|-------------|----------------|
| **`can_call()`** | Native multi-agent collaboration | Menu Creator calls recipe agents directly |
| **`Memory`** | Conversation context retention | Info Checker remembers previous interactions |
| **`stream_invoke()`** | Real-time streaming progress | Watch agents work step by step |
| **`a_stream_invoke()`** | Async streaming | Non-blocking async execution |
| **`planning_interval`** | Multi-step planning | Menu Creator plans every 3 steps |
| **`max_steps`** | Execution control | Prevents infinite loops |
| **`terminate_on_text`** | Termination control | Stops when plain text is returned |
| **`tool_choice`** | Fine-grained tool control | Forces tool usage when needed |
| **`DuckDuckGoSearchTool`** | Web search | Discover new recipes online |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    🎄 ORCHESTRATOR AGENT 🎄                      │
│    Coordinates agents using can_call() multi-agent support      │
│         Supports streaming & async execution modes              │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     INFO CHECKER AGENT                          │
│     Validates user requirements (guests, allergies, diet)       │
│              💾 Uses Memory for conversation context            │
└─────────────────────────────────────────────────────────────────┘
                                │
            ┌───────────────────┼───────────────────┬──────────────┐
            ▼                   ▼                   ▼              ▼
┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐
│  APPETIZER AGENT  │ │  MAIN DISH AGENT  │ │SECOND PLATE AGENT │ │  DESSERT AGENT    │
│   RAG Search 🥗   │ │   RAG Search 🍝   │ │   RAG Search 🥩   │ │   RAG Search 🍰   │
│  tool_choice ✓    │ │  tool_choice ✓    │ │  tool_choice ✓    │ │  tool_choice ✓    │
└───────────────────┘ └───────────────────┘ └───────────────────┘ └───────────────────┘
            │                   │                   │              │
            └───────────────────┴───────────────────┴──────────────┘
                                │
                        ┌───────┴───────┐
                        ▼               ▼
┌───────────────────────────────┐ ┌─────────────────────────────────┐
│     RECIPE RESEARCH AGENT     │ │        MENU CREATOR AGENT       │
│    🌐 DuckDuckGoSearchTool    │ │   📊 planning_interval=3        │
│    Discovers recipes online   │ │   🤖 can_call() all agents      │
└───────────────────────────────┘ │   Compiles balanced menu        │
                                  └─────────────────────────────────┘
```

## 📂 Project Structure

```
datapizza-ai/
├── agents/
│   ├── __init__.py
│   ├── info_checker.py      # Validates user preferences (with Memory)
│   ├── recipe_agents.py     # Appetizer, Main, Second, Dessert agents (inherit from Agent)
│   ├── menu_creator.py      # Final menu compilation (with planning_interval)
│   └── orchestrator.py      # Main coordination (can_call, streaming, async)
├── config/
│   ├── __init__.py
│   └── settings.py          # Configuration settings
├── database/
│   ├── __init__.py
│   ├── vector_store.py      # ChromaDB vector store
│   └── recipe_loader.py     # Recipe data loader
├── models/
│   ├── __init__.py
│   ├── recipe.py            # Recipe model
│   ├── menu.py              # Menu model
│   └── user_preferences.py  # User preferences model
├── tools/
│   ├── __init__.py
│   └── recipe_search.py     # RAG search tools
├── data/                    # Vector DB storage (auto-created)
├── main.py                  # Entry point
├── requirements.txt
└── README.md
```

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   cd datapizza-ai
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   ```bash
   # Create .env file
   echo "OPENAI_API_KEY=your-api-key-here" > .env
   ```

## 📖 Usage

### Interactive Mode

```bash
python main.py
```

Then describe your Christmas dinner requirements:
```
Your request: I need a menu for 8 people, 2 are vegetarian, avoid nuts
```

### Interactive Commands

| Command | Description |
|---------|-------------|
| `help` | Show help message |
| `quit` / `exit` | Exit the program |
| `reload` | Reload the recipe database |
| `stream` | Toggle streaming mode (real-time progress) |
| `native` | Toggle native multi-agent mode (can_call) |
| `async` | Toggle async streaming mode |
| `web <query>` | Search web for recipes |
| `memory` | Show conversation memory |
| `clear` | Clear conversation memory |

### Single Request Mode

```bash
# Standard mode
python main.py "Christmas dinner for 6 guests, one vegan, prefer traditional"

# With streaming progress
python main.py --stream "Christmas dinner for 8 people"

# With native multi-agent mode
python main.py --native "Menu for 10 guests, 2 vegetarian"
```

### Programmatic Usage

```python
from agents.orchestrator import ChristmasMenuOrchestrator

# Create orchestrator
orchestrator = ChristmasMenuOrchestrator()

# Standard mode
result = orchestrator.run(
    "I need a Christmas menu for 10 people. "
    "3 guests are vegetarian, 1 is vegan."
)

if result["success"]:
    print(result["menu"])

# Native multi-agent mode (using can_call)
result = orchestrator.run_with_native_agents(
    "Christmas dinner for 8 guests, 2 vegetarian"
)

# Streaming mode
for update in orchestrator.run_streaming("Menu for 6 people"):
    print(update["message"])

# Async streaming mode
import asyncio

async def main():
    async for update in orchestrator.run_async_streaming("Menu for 8"):
        print(update["message"])

asyncio.run(main())

# Web search for recipes
results = orchestrator.search_web_recipes("vegan Christmas dessert")
print(results)
```

### Using Memory for Iterative Refinement

```python
from agents.orchestrator import ChristmasMenuOrchestrator

orchestrator = ChristmasMenuOrchestrator()

# First request
result1 = orchestrator.info_checker.extract_preferences(
    "I need a menu for 8 people"
)

# Add more details (Memory remembers the context!)
result2 = orchestrator.info_checker.update_preferences(
    "Actually, 2 of them are vegetarian and one has a nut allergy"
)

# Check memory
print(orchestrator.info_checker.get_memory_summary())

# Clear memory for fresh start
orchestrator.clear_info_checker_memory()
```

## 🛠️ Customization

### Adding Custom Recipes

Create a JSON file with your recipes:

```json
{
  "recipes": [
    {
      "id": "custom_001",
      "name": "My Special Dish",
      "description": "A family favorite",
      "category": "main_dish",
      "ingredients": ["ingredient1", "ingredient2"],
      "instructions": ["Step 1", "Step 2"],
      "dietary_tags": ["vegetarian"],
      "allergens": ["dairy"],
      "is_christmas_traditional": true
    }
  ]
}
```

Load it:
```python
orchestrator.reload_recipes("path/to/recipes.json")
```

### Recipe Categories

- `appetizer` - Antipasti
- `main_dish` - Primi piatti (pasta, risotto, soup)
- `second_plate` - Secondi piatti (meat, fish, main protein)
- `dessert` - Dolci

### Dietary Tags

- `vegan`
- `vegetarian`
- `gluten_free`
- `dairy_free`
- `nut_free`
- `traditional`

## 🔧 Configuration

Edit `config/settings.py` or use environment variables:

### LLM Provider Selection

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `openai` | LLM provider: `openai` or `ollama` |

### OpenAI Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | - | Your OpenAI API key |
| `DEFAULT_MODEL` | `gpt-4o-mini` | OpenAI model to use |

### Ollama Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.1:8b` | Ollama model to use |

**Recommended Ollama models:**
- `llama3.1:8b` - Good balance of speed and quality (default)
- `llama3.1:70b` - Higher quality, needs more resources
- `mistral:7b` - Fast and capable
- `mixtral:8x7b` - High quality mixture of experts
- `qwen2.5:7b` - Good for instructions
- `gemma2:9b` - Good for chat

### Other Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | Vector DB storage path |
| `TEMPERATURE` | `0.7` | Agent creativity level |

### Using Ollama

1. **Install Ollama:**
   ```bash
   # macOS/Linux
   curl -fsSL https://ollama.com/install.sh | sh
   
   # Windows - download from https://ollama.com
   ```

2. **Pull a model:**
   ```bash
   ollama pull llama3.1:8b
   ```

3. **Start Ollama server:**
   ```bash
   ollama serve
   ```

4. **Configure the app:**
   ```bash
   export LLM_PROVIDER=ollama
   export OLLAMA_MODEL=llama3.1:8b
   python main.py
   ```

   Or use the CLI flag:
   ```bash
   python main.py --ollama "Christmas dinner for 8"
   ```

## 📊 How It Works

1. **Info Checker Agent** (with Memory) analyzes your request to extract:
   - Number of guests
   - Dietary restrictions (vegan, vegetarian)
   - Food allergies
   - Recipe preferences
   - *Remembers context for iterative refinement*

2. **Recipe Search Agents** use RAG with `tool_choice` to find:
   - Matching appetizers
   - Suitable main dishes
   - Appropriate second plates
   - Fitting desserts
   - *Optional: web search for new recipes*

3. **Menu Creator Agent** (with `planning_interval`) combines suggestions into:
   - A balanced, cohesive menu
   - Wine pairing recommendations
   - Preparation timeline
   - Shopping tips
   - *Plans systematically using multi-step planning*

## 🎯 Datapizza AI Feature Examples

### Native Multi-Agent with can_call()

```python
# The menu creator can directly call recipe agents
menu_creator.can_call([
    appetizer_agent,
    main_dish_agent,
    second_plate_agent,
    dessert_agent,
    recipe_researcher,  # Web search agent
])

# Single call handles all coordination
result = menu_creator.create_menu_with_agents(preferences)
```

### Planning Interval

```python
# Menu creator plans every 3 steps
agent = Agent(
    name="menu_creator",
    planning_interval=3,
    planning_prompt="Review progress and plan next steps..."
)
```

### Tool Choice Control

```python
# Force tool usage on first call
response = agent.run(prompt, tool_choice="required_first")

# Specify which tool to use
response = agent.run(prompt, tool_choice=["search_appetizers"])
```

## 🧪 Testing

### Run Test Notebook

A comprehensive Jupyter notebook is available to test all agents:

```bash
cd notebooks
jupyter notebook test_agents.ipynb
```

The notebook tests:
- ✅ InfoCheckerAgent (with Memory)
- ✅ AppetizerAgent, MainDishAgent, SecondPlateAgent, DessertAgent
- ✅ MenuCreatorAgent (with planning_interval)
- ✅ RecipeResearchAgent (DuckDuckGo web search)
- ✅ ChristmasMenuOrchestrator (full pipeline, streaming, native mode)
- ✅ Provider switching (OpenAI ↔ Ollama)
- ✅ Vector Store & Recipe Database
- ✅ Async execution

### Run Unit Tests

```bash
pytest tests/
```

## 📝 Example Output

```
╔══════════════════════════════════════════════════════════════════╗
║                    🎄 Christmas Dinner 2024 🎄                   ║
║                      For 8 Guests                                ║
╚══════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🥗 APPETIZERS (Antipasti)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Bruschetta al Pomodoro (vegan)
• Smoked Salmon Canapés

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🍝 MAIN DISHES (Primi Piatti)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Traditional Christmas Lasagna
• Vegan Mushroom Risotto (vegan)

...
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

MIT License - feel free to use this project for your Christmas dinner planning!

---

**Buon Natale! 🎅🎄**
