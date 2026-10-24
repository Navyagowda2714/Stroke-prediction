# Lightweight Python image
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so Docker can cache this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app, trained model and scaler
COPY . .

# Optional: set GOOGLE_API_KEY at runtime to enable the Gemini explanation
# docker run -e GOOGLE_API_KEY=... -p 5000:5000 stroke-prediction
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "app:app"]
