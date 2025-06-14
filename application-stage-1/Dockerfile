FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY cert/ /etc/ssl/certs
COPY app/ ./app/
COPY documents/ ./documents/
CMD uvicorn app.main:app --host ${API_HOST:-0.0.0.0} --port ${API_PORT:-8000}
