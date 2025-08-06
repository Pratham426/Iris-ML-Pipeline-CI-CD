# Base Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY req.txt .
RUN pip install --no-cache-dir -r req.txt

# Copy all other files
COPY . .

# Create directory for artifacts
RUN mkdir -p models artifacts

# Copy trained model and artifacts
COPY models/iris_model.pkl models/
COPY artifacts/ artifacts/

# Expose port
EXPOSE 5000

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]