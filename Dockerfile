# Use a slim Python 3.11 base image
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required for building some packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Run the dataset and model building scripts
RUN python main.py

# Expose Streamlit default port
EXPOSE 8501

# Add a healthcheck rule
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Start the Streamlit application using the root entrypoint
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
