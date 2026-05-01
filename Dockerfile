FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt requirements-dev.txt pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests
COPY data/fixtures ./data/fixtures

RUN python -m pip install --upgrade pip && \
    pip install -r requirements-dev.txt && \
    pip install -e .

CMD ["bash", "scripts/ci_smoke.sh"]
