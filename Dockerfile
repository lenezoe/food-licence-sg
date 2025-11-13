# -------------------------------
# Stage 1: Base environment
# -------------------------------
FROM python:3.10-slim

# Prevent Python from writing .pyc files and buffering output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory in container
WORKDIR /app

# -------------------------------
# Stage 2: Install dependencies
# -------------------------------
# Install system dependencies required by FAISS and numpy
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt first (for efficient caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------------
# Stage 3: Copy application code
# -------------------------------
COPY . .

# Expose Streamlit’s default port
EXPOSE 8501

# Set Streamlit environment variables
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ENABLECORS=false
ENV STREAMLIT_SERVER_ENABLEXSRSFPROTECTION=false

# -------------------------------
# Stage 4: Run the app
# -------------------------------
CMD ["streamlit", "run", "Licence Helper.py"]
