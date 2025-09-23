# 1. Base image - Matches python = "^3.9" from your pyproject.toml
FROM python:3.12-slim

# 2. Environment Variables
ENV PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.8.2 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false

# Add Poetry to PATH
ENV PATH="$POETRY_HOME/bin:$PATH"

# 3. Install system dependencies
# ffmpeg is required for openai-whisper and pydub
# curl is used to download the Poetry installer
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg curl && \
    # Clean up apt caches to reduce image size
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 4. Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# 5. Set working directory
WORKDIR /app

# 6. Copy dependency definition files
# Copy pyproject.toml and poetry.lock (if it exists).
# It's highly recommended to commit poetry.lock to your repository for reproducible builds.
COPY pyproject.toml /app/pyproject.toml
COPY poetry.lock /app/poetry.lock

# 7. Install project dependencies using Poetry
# This step leverages Docker layer caching. It will only re-run if
# pyproject.toml or poetry.lock changes.
# --no-root: Do not install the project package itself.
# --no-dev: Skip development dependencies.
# Ensure your pyproject.toml lists all necessary dependencies like gradio, openai-whisper, etc.
RUN poetry install --no-interaction --no-ansi --no-root --no-dev

# 8. Copy application code
# This assumes your source code is in a 'src' directory in your project root.
COPY ./src /app/src
COPY ./models /app/models
# 9. Expose the port Gradio runs on (default is 7860)
EXPOSE 7860

# 10. Define the command to run the application
# The application (app.py) loads OPENAI_API_KEY from the environment.
# You'll need to pass this environment variable when running the container, e.g.:
# docker run -e OPENAI_API_KEY='your_actual_api_key' -p 7860:7860 <your_image_name>
CMD ["python", "app.py"] 