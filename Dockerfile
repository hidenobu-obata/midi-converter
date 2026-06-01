FROM python:3.11-slim

# システムの更新とffmpegの確実なインストール
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 依存ライブラリを直接インストール（requirements.txtを無視）
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir flask gunicorn tensorflow basic-pitch Werkzeug

COPY . /app

# ポート5000番で、タイムアウトを5分に延ばして起動
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--timeout", "300", "--workers", "1"]