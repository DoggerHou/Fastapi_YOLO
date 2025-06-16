# Базовый образ с поддержкой CUDA и PyTorch
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# Установка системных библиотек для OpenCV и видео
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Установка Python-зависимостей
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Копирование остального проекта
COPY . .

# Указание порта для FastAPI
EXPOSE 8000

# Запуск FastAPI через uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
