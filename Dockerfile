FROM python:3.11-slim

# Minimal OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app
COPY . /app

# Streamlit must bind to 0.0.0.0 and use the platform port
ENV PORT=8080
EXPOSE 8080
CMD ["streamlit", "run", "app_llama_rag.py", "--server.port=8080", "--server.address=0.0.0.0"]
