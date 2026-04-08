# Use official Python 3.11 slim image
FROM python:3.11-slim

# Build timestamp — force rebuild: 2026-04-08-v2
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Force fresh copy of all files
COPY app/ ./app/
COPY baseline/ ./baseline/
COPY inference.py .
COPY openenv.yaml .
COPY README.md .

EXPOSE 7860

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "7860"]