# 1. 最初からPython 3.11と必要なシステムが入っているベース環境を使用
FROM python:3.11-slim

# 2. Renderの書き込み禁止を無視して、システムにffmpegを確実に叩き込む
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# 3. 作業するフォルダを決める
WORKDIR /app

# 4. あなたのプログラム一式を丸ごとコピー
COPY . /app

# 5. エラーの元凶だったライブラリは無視して、強制的に必要なものだけ入れる
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir gunicorn flask basic-pitch tensorflow

# 6. アプリを起動する（タイムアウトもここで5分に設定）
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--timeout", "300", "--workers", "1"]