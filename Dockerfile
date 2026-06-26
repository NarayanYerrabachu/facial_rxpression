FROM python:3.14-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install pipenv

WORKDIR /app

COPY Pipfile ./
RUN pipenv lock && pipenv install --system

COPY run.py ./
COPY audio_portrait/ ./audio_portrait/

RUN mkdir -p assets output

CMD ["python", "run.py", "--help"]
