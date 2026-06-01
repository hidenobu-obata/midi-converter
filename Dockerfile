FROM python:3.11-slim

# システムの更新とffmpegのインストール
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# TensorFlowとBasic Pitchの大食いを抑える魔法の環境変数
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV OMP_NUM_THREADS=1
ENV TF_NUM_INTRAOP_THREADS=1
ENV TF_NUM_INTEROP_THREADS=1

# 依存ライブラリのインストール
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir flask gunicorn tensorflow-cpu basic-pitch Werkzeug

COPY . /app

# ポート10000番、タイムアウト5分、1ワーカーで起動
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000", "--timeout", "300", "--workers", "1", "--threads", "1"]