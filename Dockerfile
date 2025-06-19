# Используем легкий образ Python
FROM python:3.10-slim

# Устанавливаем зависимости
RUN apt-get update && apt-get install -y --no-install-recommends gcc python3-dev && rm -rf /var/lib/apt/lists/*

# Рабочая директория
WORKDIR /app

# Копируем зависимости и устанавливаем их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект
COPY . .

# Запускаем Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "app:app"]

# Открываем порт
EXPOSE 8080
