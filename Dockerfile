# takes about 20 ~ 25 minutes
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# 1. 從官方映像檔複製 uv 執行檔 (這是最乾淨的安裝方式)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# 設定環境變數
ENV DEBIAN_FRONTEND=noninteractive
# 設定 uv 的編譯緩存目錄和 Python 安裝目錄，避免權限問題
ENV UV_CACHE_DIR=/root/.cache/uv
ENV UV_PYTHON_INSTALL_DIR=/usr/local/python

# 2. 只需要安裝最基礎的系統工具 (不需要 python3-pip 或 deadsnakes)
RUN apt-get update && apt-get install -y \
    curl \
    tmux \
    git \
    && rm -rf /var/lib/apt/lists/*

# 3. 讓 uv 安裝 Python 3.12 並建立虛擬環境
# 這裡我們將虛擬環境建立在 /app/.venv
WORKDIR /app
COPY FastChat /app/build/FastChat
COPY nano-PEARL /app/build/nano-PEARL

RUN uv python install 3.12 && \
    uv venv .venv --python 3.12

# 4. 重要：將虛擬環境加入 PATH
# 這樣之後的指令 (python, pip) 就會自動使用 .venv 裡的，而不用 source activate
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN python3 -m ensurepip --default-pip \
    && python3 -m pip install torch transformers accelerate psutil packaging setuptools ninja
RUN python3 -m pip install flash-attn --no-build-isolation
RUN python3 -m pip install -e build/nano-PEARL
RUN python3 -m pip install -e build/FastChat

# Install mini-sglang (reference only, not modified)
COPY mini-sglang /app/build/mini-sglang
RUN python3 -m pip install -e build/mini-sglang

# Install mini-sglang-pearl integration  
COPY mini-sglang-pearl /app/build/mini-sglang-pearl
RUN python3 -m pip install -e build/mini-sglang-pearl

RUN rm -rf /app/build

CMD ["bash"]
