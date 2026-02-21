# 1. Use an official lightweight Python image
FROM python:3.10-slim

# 2. Set working directory
WORKDIR /app

# 3. Copy project files
COPY . /app

# 4. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Set environment variables (optional)
ENV MLFLOW_TRACKING_URI=http://localhost:5000

# 6. Run MLflow model serving
CMD ["mlflow", "models", "serve", "-m", "runs:/<RUN_ID>/model", "-h", "0.0.0.0", "-p", "5000", "--no-conda"]
