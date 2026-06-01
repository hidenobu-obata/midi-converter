FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir flask gunicorn tensorflow basic-pitch Werkzeug

COPY . /app

# 👇 ポートを 5000 から 10000 に変更！
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000", "--timeout", "300", "--workers", "1"]