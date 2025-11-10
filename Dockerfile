# 1.1 - Base Python 3.11-slim image
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN pip install uv
COPY pyproject.toml uv.lock ./
ENV UV_HTTP_TIMEOUT=300
RUN uv sync --frozen --no-dev


COPY . .

ENV PYTHONPATH=/app:/app/app

EXPOSE 8000

CMD ["uv", "run", "python", "app/main.py"]
