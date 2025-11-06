# API de Classificação de Depósito Bancário (MLOps)

## Visão Geral

Esta é uma API RESTful em **FastAPI** para servir um modelo de **Machine Learning** (Random Forest) que prevê a probabilidade de um cliente subscrever um depósito bancário. O projeto está contêinerizado com **Docker** para garantir portabilidade total (MLOps).

### Stack Tecnológica

- **API:** FastAPI, Uvicorn
- **ML:** Scikit-learn, Pandas, joblib
- **Deploy:** Docker

## Estrutura Essencial
  .        
  ├── app/        
  │ └── main.py        
  ├── Dockerfile        
  ├── .gitignore        
  ├── LICENSE        
  ├── requirements.txt        
  └── run_server.py      

## Como Rodar a Aplicação com Docker

### Pré-requisitos
Certifique-se de que o Docker Desktop esteja em execução.

### 1. Construir a Imagem

Na raiz do projeto (`api-deposito-bancario/`), execute:

```bash
docker build -t api-deposito-bancario:latest .
```
### 2. Executar o Contêiner

Execute a imagem, mapeando a porta local(`8000`) para a porta do contêiner:
```bash
docker run -d --name deposito-api -p 8000:8000 api-deposito-bancario:latest
```

## Acesso e Teste

A API está ativa em segundo plano.

### Documentação (Swagger UI)

Acesse a documentação interativa para testar o endpoint `/predict`:

[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## Nota MLOps

A API utiliza o `run_server.py` como ponto de entrada `(__main__)` para resolver o problema de carregamento de classes customizadas do `joblib` no ambiente Docker, garantindo o deploy estável.

---
