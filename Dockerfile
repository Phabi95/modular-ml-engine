FROM ghcr.io/astral-sh/uv:python3.9-bookworm-slim
WORKDIR /app/svm_project
COPY pyproject.toml .
RUN uv sync
COPY src/ src/
CMD ["uv", "run", "python", "-m", "src.main"]