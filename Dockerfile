# Use a lightweight python base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

# Set working directory
WORKDIR /code

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements to the container
COPY ./app/backend/requirements.txt /code/requirements.txt

# Install PyTorch CPU first (to keep container lightweight and build fast),
# then install remaining backend dependencies from requirements.txt
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /code/requirements.txt

# Copy all project files to the container
COPY . /code/

# Expose the default port (7860 is used by Hugging Face Spaces by default)
EXPOSE 7860

# Run the FastAPI application using uvicorn
CMD ["sh", "-c", "uvicorn app.backend.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
