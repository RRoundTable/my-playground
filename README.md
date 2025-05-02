# Title Evaluator

A LangChain and LangGraph-based agent that evaluates titles based on their content.

## Setup

1. Install Poetry if you haven't already:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Clone this repository and install dependencies:
```bash
poetry install
```

3. Create a `.env` file in the project root with your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

The title evaluator can be used in two ways:

1. As a standalone script:
```bash
poetry run python title_evaluator.py
```

2. Imported as a module:
```python
from title_evaluator import evaluate_title_agent

evaluation = evaluate_title_agent(
    title="Your Title",
    content="Your content here..."
)
print(evaluation)
```

## Features

The evaluator considers:
- Relevance to the content
- Clarity and conciseness
- Engagement and appeal
- SEO-friendliness

## Requirements

- Python 3.9+
- OpenAI API key
- Poetry for dependency management

## Commnads
```bash
make init  # then, open http://hostname:9000/
make down

STACK_NAME=$SERVICE_NAME make service-up   # requires .env
STACK_NAME=$SERVICE_NAME make service-down # requires .env
```

## Adding Stacks
1. Open portainer
2. Stack -> Add stack
3. Repository
4. Add stack inforamtion
5. Deploy

## Service Stacks
- `init.yaml`
    - portainer
- `network.yaml`
    - traefik
    - cloudflare-companion
    - cloudflare-ddns
- `n8n.yaml`
    - n8n
- `nocodb.yaml`
    - nocodb
    - nocodb-db (postgresql)
