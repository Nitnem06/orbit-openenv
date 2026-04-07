# Use official Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Expose port 7860 (Hugging Face Spaces default)
EXPOSE 7860

# Run the FastAPI server with uvicorn
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "7860"]