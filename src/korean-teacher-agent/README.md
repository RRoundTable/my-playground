# Korean Teacher Agent

A LangChain and LangGraph based agent system for evaluating and enhancing Korean language learning content.

## Features

- Notion API integration
- Title evaluation for Korean language learning videos
- Commenting on Notion pages and blocks
- Structured agent design using LangGraph

## Project Structure

```
src/
├── agents/           # Agent implementations
├── clients/          # API client code
├── prompts/          # Prompt templates
├── tools/            # Tool implementations
│   ├── notion_tools.py    # Notion API tools
│   ├── title_tools.py     # Title evaluation tools
│   └── tools_registry.py  # Aggregates all tools
└── main.py           # Entry point
```

## Prerequisites

- Python 3.10+
- Poetry package manager
- Notion API key
- OpenAI API key

## Installation

1. Clone the repository

2. Install dependencies using Poetry:

```bash
cd korean-teacher-agent
poetry install
```

## Configuration

Create a `.env` file in the project root with the following environment variables:

```
OPENAI_API_KEY=your_openai_api_key
NOTION_TOKEN=your_notion_api_key
PHOENIX_COLLECTOR_ENDPOINT=your_phoenix_endpoint  # Optional
PHOENIX_API_KEY=your_phoenix_api_key              # Optional
```

## Usage

```python
from src.agents.notion_agent import run_notion_agent

# Run the agent with a query
response = run_notion_agent("page id 1e9ff0df28478038a184fe3371797f96에 title을 평가한 후 평가내용을 노션 댓글로 남겨줘")
print(response)
```

## Development

To add new tools:

1. Create a new file in the `tools` directory
2. Implement your tool functions using the `@tool` decorator
3. Add your tools to the appropriate exports list
4. Import and register your tools in `tools_registry.py`

## License

[MIT License](LICENSE)
