# Use official Python image as base
FROM python:3.10-slim

# Install curl
RUN apt install -y curl

# Copy application files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download the new version model
RUN curl -X GET -o current.bst "https://storage.googleapis.com/tymestack-artifacts/housing-prediction/current.bst"

# Expose the port for FastAPI
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
