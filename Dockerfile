# Use official Python 3.11 slim image
FROM python:3.11-slim

# Build timestamp — force rebuild: 2026-04-08-v3
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY app/ ./app/
COPY baseline/ ./baseline/
COPY server/ ./server/
COPY inference.py .
COPY openenv.yaml .
COPY README.md .
COPY pyproject.toml .

EXPOSE 7860

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "7860"]