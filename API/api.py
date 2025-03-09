from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import sys
import os

# Adiciona o diretório raiz ao PYTHONPATH (se necessário)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Carregar o modelo treinado
model_path = "API/models/random_forest_model.pkl"
try:
    loaded_model = joblib.load(model_path)
except Exception as e:
    raise RuntimeError(f"Erro ao carregar o modelo: {e}")

# Inicializa a API FastAPI
app = FastAPI()

# Modelo de dados esperado na requisição (5 campos)
class PredictionInput(BaseModel):
    iss_retention: str  # "Aplicável" ou "Não Aplicável"
    inss_tax_rate: float
    csll_tax_rate: float
    calculated_value: float
    cofins_tax_rate: float

@app.post("/predict/")
def get_prediction(payload: PredictionInput):
    """
    Endpoint para prever a probabilidade de fraude.
    """
    try:
        # Converter iss_retention para valor numérico
        if payload.iss_retention == "Aplicável":
            iss_retention_value = 1
        elif payload.iss_retention == "Não Aplicável":
            iss_retention_value = 0
        else:
            raise HTTPException(
                status_code=400,
                detail="Valor inválido para iss_retention. Use 'Aplicável' ou 'Não Aplicável'."
            )

        # Garantir que as taxas estejam entre 0 e 100
        inss_tax_rate = min(max(payload.inss_tax_rate, 0), 100)
        csll_tax_rate = min(max(payload.csll_tax_rate, 0), 100)
        cofins_tax_rate = min(max(payload.cofins_tax_rate, 0), 100)
        calculated_value = payload.calculated_value

        # Construir o array de entrada (2D) com 5 features
        X_input = [[iss_retention_value, inss_tax_rate, csll_tax_rate, calculated_value, cofins_tax_rate]]
        
        # Fazer a predição
        y_pred = loaded_model.predict(X_input)
        
        # Retornar o resultado
        return {"prediction": int(y_pred[0])}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar a previsão: {str(e)}")

@app.get("/test/")
def test():
    """
    Endpoint para testar se a API está rodando.
    """
    return {"message": "API está funcionando corretamente!"}
