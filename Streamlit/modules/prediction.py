import streamlit as st
import requests

API_URL = "http://localhost:8000/predict/"  # Atualizado para o localhost

def prediction():
    """
    Aba de Previsão de Fraude.
    Faz uma requisição à API para prever a probabilidade de fraude.
    """

    st.markdown("<h3 style='text-align: center;'>Detecção de Notas Fiscais com Impostos Incorretos</h3>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center;'>Utilize a API para classificar notas fiscais</h6>", unsafe_allow_html=True)

    # Formulário para inserção dos dados
    with st.form("fraud_prediction_form"):
        # Seleção do campo iss_retention como string
        iss_retention = st.selectbox("Retenção de ISS", options=["Aplicável", "Não Aplicável"])

        # Slider para as taxas (valores float)
        inss_tax_rate = st.slider("Taxa de INSS aplicada, se for o caso", 
                                  min_value=0.0, max_value=100.0, value=0.0, step=0.1)
        
        csll_tax_rate = st.slider("Taxa de CSLL aplicada, se for o caso", 
                                  min_value=0.0, max_value=100.0, value=0.0, step=0.1)
        
        cofins_tax_rate = st.slider("Taxa de COFINS (Contribuição para o Financiamento da Seguridade Social)", 
                                    min_value=0.0, max_value=100.0, value=0.0, step=0.1)
        
        # Valor total calculado da nota fiscal
        calculated_value = st.number_input("Valor total calculado da nota fiscal, incluindo impostos", 
                                           value=0.0, format="%.2f")
        
        submit_button = st.form_submit_button("Fazer Previsão")

    if submit_button:
        # Criar o JSON com os dados do usuário, conforme os 5 campos esperados pela API
        payload = {
            "iss_retention": iss_retention,
            "inss_tax_rate": inss_tax_rate,
            "csll_tax_rate": csll_tax_rate,
            "calculated_value": calculated_value,
            "cofins_tax_rate": cofins_tax_rate
        }

        # Fazer a requisição à API
        try:
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()  # Levanta erro para status HTTP 4xx ou 5xx

            result = response.json()  # Converte resposta para JSON

            # Verificar se a chave 'prediction' existe na resposta
            if "prediction" in result:
                st.subheader("Resultado da Previsão")
                if result["prediction"] == 1:
                    st.write("<span style='color:red; font-size:24px;'>Atenção: Nota Inválida.</span>", unsafe_allow_html=True)
                else:
                    st.write("<span style='color:green; font-size:24px;'>Seguro: Nota Válida.</span>", unsafe_allow_html=True)
            else:
                st.error("Erro: A resposta da API não contém a chave 'prediction'. Verifique os logs da API.")
        except requests.exceptions.RequestException as e:
            st.error(f"Erro na requisição: {e}")

# Chama a função para exibir a aba
prediction()
