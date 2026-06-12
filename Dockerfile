FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

ARG UID=1000
ARG GID=1000
RUN groupadd --gid "${GID}" tempora \
    && useradd --uid "${UID}" --gid "${GID}" --create-home tempora

COPY pyproject.toml README.md ./
COPY requirements ./requirements
COPY src ./src
COPY tests ./tests
COPY docs ./docs

RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements/docker-cpu.txt \
    && python -m pip install -e ".[dev]" --no-deps \
    && chown -R tempora:tempora /workspace

USER tempora

CMD ["bash", "-lc", "python -m pytest && python -m ruff check . && python -m ruff format --check . && python -m mypy src"]
