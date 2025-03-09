import streamlit as st
import pandas as pd
from modules.data_loading import load_data
from modules.eda import exploratory_data_analysis
from modules.preprocessing import data_preprocessing
from modules.modeling import model_training_and_evaluation
from modules.prediction import prediction  

# Configuração inicial da página
st.set_page_config(
    page_title="Detecção de Notas Fiscais Fraudulentas",
    page_icon="📊",
    layout="wide"
)

# Carregar dados
df = load_data()

# Formatar apenas as colunas numéricas globalmente (opcional)
pd.options.display.float_format = '{:.2f}'.format

# Título principal
st.title("Análise de Notas Fiscais com Impostos Incorretos")
# st.markdown("---")

# menu = st.sidebar.selectbox(
#     "Escolha a seção",
#     ["Utilizar Modelo para Predições", "Tratamento dos Dados e Criação do Modelo"]
# )
tab1, tab2, tab3, tab4 = st.tabs(["📊 Análise Exploratória dos Dados", "⚙️ Pré-processamento dos Dados", "🤖 Modelagem e Avaliação dos Modelos", "🔍 Classificar Nota"])
with tab1:
    # 📁 Seção 1: Análise Exploratória dos Dados (EDA)
    exploratory_data_analysis(df)
with tab2:
    # 🛠️ Seção 2: Pré-processamento dos Dados
    df = data_preprocessing(df)
with tab3:
    # 🤖 Seção 3: Modelagem e Avaliação dos Modelos
    rf_classifier, preprocessor = model_training_and_evaluation(df)
with tab4:
    # 📉 Seção 4: Previsão de Fraude
    prediction()

# Rodapé
st.markdown("---")
st.write("© 2025 - Detecção de Notas Fiscais com Impostos Incorretos")