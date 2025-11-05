# Imagem base Python
FROM python:3.12-slim

# Diretório de trabalho
WORKDIR /app

# Copia arquivos de configuração de dependencias
COPY requirements.txt .

# Instala as dependencias
ENV PYTHONPATH="/app"
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código da API e o modelo
COPY app/ app/
COPY src/ src/
COPY models/ models/

# Porta utilizada pelo uvicorn
EXPOSE 8000

# Roda a aplicação Uvicorn
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]