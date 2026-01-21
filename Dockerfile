FROM python:3.13-slim

WORKDIR /app

# instala deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copia o projeto
COPY . .

# expõe porta
EXPOSE 8000

# inicia a API
CMD ["gunicorn", "app.main:app", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "120"]
