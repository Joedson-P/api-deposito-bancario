import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal

# --- Configuração e carregamento do modelo ---

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'rf_pipeline.pkl')

MODEL = None
try:
    MODEL = joblib.load(MODEL_PATH)
    print("Modelo 'rf_pipeline.pkl' carregado com sucesso.")
except FileNotFoundError:
    print(f"ERRO CRÍTICO: Arquivo de modelo não encontrado em {MODEL_PATH}")
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")

# --- DEfinição do schema de entrada ---

# pydantic define a estrutura, tipo e validação dos dados de entrada da API
class InputFeatures(BaseModel):
    # DADOS NUMÉRICOS
    age: int = Field(..., ge = 18, le = 120, description = 'Idade do cliente.')
    balance: float = Field(..., description = "Balanço anual médio.")
    duration: int = Field(..., ge = 0, description = "Duração do último contato em segundos.")
    campaign: int = Field(..., ge = 1, description = "Número de contatos durante esta campanha.")
    previous: int = Field(..., ge = 0, description = "Número de contatos antes desta campanha")

    # DADOS CATEGÓRICOS
    job: Literal['management', 'technician', 'entrepreneur', 'blue-collar', 'unknown', 'retired', 'admin.', 'services', 'self-employed', 'unemployed', 'housemaid', 'student']
    marital: Literal['married', 'single', 'divorced']
    education: Literal['tertiary', 'secondary', 'unknown', 'primary']
    default: Literal['no', 'yes']
    housing: Literal['no', 'yes']
    loan: Literal['no', 'yes']
    contact: Literal['unknown', 'cellular', 'telephone']
    month: Literal['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    poutcome: Literal['unknown', 'other', 'failure', 'success']

# --- Inicialização da API ---    

app = FastAPI(
    title = "API de Previsão de Depósito Bancário",
    version = '1.0.0',
    description = "Serviço REST para prever a subscrição de depósitos a prazo (YES/NO)."
)

# --- Endpoints da API ---

@app.get("/")
def home():
    """Endpoint de saúde/status da API."""
    return {"status": "ok", "model_loaded": MODEL is not None, "api_version": "1.0.0"}
@app.post("/predict")
def predict(features: InputFeatures):
    """
    Endpoint principal para prever a subscrição de depósito.
    Recebe um JSON com as features do cliente e retorna a probabilidade.
    """
    if MODEL is None:
        raise HTTPException(status_code = 500, detail = "Modelo não carregado ou erro de inicialização.")
    
    # Convertendo o objeto pydantic em um dicionário
    input_data_dict = features.model_dump()

    # Convertendo o dicionário para o formato esperado pela pipeline (DataFrame)
    input_df = pd.DataFrame([input_data_dict])

    # Adição das colunas faltantes no DF
    input_df['day'] = 1  
    input_df['pdays'] = -1 

    # Fazendo a previsão das probabilidades
    probas = MODEL.predict_proba(input_df)[0]

    prob_no = probas[0]
    prob_yes = probas[1]

    return {
        "prediction_probability": {
            "no": prob_no,
            "yes": prob_yes
        },
        "threshold_used": 0.6239,
        "description": "Probabilidade de subscrever ('yes') ou não subscrever ('no') o depósito."
    }