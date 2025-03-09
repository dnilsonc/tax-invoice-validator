# Projeto de Detecção de Notas Fiscais com Impostos Incorretos

Este projeto tem como objetivo detectar notas fiscais com impostos incorretos utilizando técnicas de Machine Learning. A solução é composta por duas partes principais:

1. **API** (com FastAPI) para disponibilizar o modelo de forma escalável.  
2. **[Aplicação Streamlit](https://tax-invoice-validator.streamlit.app/)** para análise de dados, visualização e testes interativos.
<img src="\img\app.png" style="height:25rem;" />

## Google Colab
Você pode acessar o notebook do projeto no [Google Colab](https://colab.research.google.com/drive/1dnBPHeDEy3yflbQbv-ZfqcXRM2wXbTZA?usp=sharing).

---

## Sumário

- [Descrição do Projeto](#descrição-do-projeto)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Instalação](#instalação)
  - [Executando a API](#executando-a-api)
  - [Executando a Aplicação Streamlit](#executando-a-aplicação-streamlit)
- [Como Usar](#como-usar)
  - [Uso da API](#uso-da-api)
- [Contribuindo](#contribuindo)
- [Licença](#licença)

---

## Descrição do Projeto

O foco do projeto é identificar notas fiscais com valores de impostos inconsistentes. O modelo de classificação (ex.: Random Forest) foi treinado para distinguir notas válidas de notas inválidas, baseado em diversas variáveis (ISS, INSS, CSLL, PIS, COFINS, entre outras).

### Principais Etapas

1. **Análise Exploratória (EDA)**: Explorar e entender o conjunto de dados.
2. **Pré-processamento**: Limpeza, tratamento de dados ausentes, engenharia de atributos.
3. **Modelagem**: Treinamento do modelo de classificação, avaliação e métricas.
4. **API**: Disponibilizar o modelo por meio de uma API (FastAPI).
5. **Interface Streamlit**: Visualizar resultados, interagir com o modelo e efetuar previsões.

---

## Estrutura do Projeto

A organização do projeto está dividida em dois diretórios principais: **API** e **Streamlit**. Além disso, há arquivos auxiliares na raiz do repositório. Segue um resumo:

```
.
├── API/
│   ├── models/
│   │   └── random_forest_model.pkl
│   ├── api.py                  # Arquivo principal da API FastAPI
│   ├── COMANDOS-RODAR-API.txt  # Instruções/Comandos para rodar a API
│   ├── Dockerfile              # Configuração para containerizar a API
│   └── requirements.txt        # Dependências necessárias para a API
│
├── Streamlit/
│   ├── modules/
│   │   ├── data_loading.py     # Carregamento e manipulação de dados
│   │   ├── eda.py              # Módulo para Análise Exploratória (EDA)
│   │   ├── modeling.py         # Modelagem e treino de modelos
│   │   ├── prediction.py       # Funções para prever com entrada do usuário
│   │   └── preprocessing.py    # Pré-processamento de dados
│   ├── app.py                  # Aplicação Streamlit para visualização e interação
│   └── requirements.txt        # Dependências necessárias para o Streamlit
│
├── README.md                   # Documentação do projeto
```

---

## Tecnologias Utilizadas

- **Python 3.9+**
- **FastAPI** (API de previsão)
- **Streamlit** (Interface Web)
- **Pandas**, **Numpy** (Manipulação de dados)
- **Scikit-learn**, **XGBoost** (Modelagem e Machine Learning)
- **Joblib** (Serialização do modelo)
- **Docker** (Opcional, para conteinerização)

---

## Instalação

### Executando a API

```bash
cd API
pip install -r requirements.txt
uvicorn api:app --reload
```
Acesse a API em: [http://127.0.0.1:8000](http://127.0.0.1:8000)

### Executando a Aplicação Streamlit

```bash
cd Streamlit
pip install -r requirements.txt
streamlit run app.py
```
Acesse a interface em: [http://localhost:8501](http://localhost:8501)

---

## Como Usar

### Uso da API

- **POST /predict/** - Envie um JSON para prever fraude:

```json
{
  "iss_retention": 5.5,
  "inss_tax_rate": 10.0,
  "csll_tax_rate": 9.0,
  "calculated_value": 10000.0,
  "cofins_tax_rate": 3.0
}
```

Resposta:

```json
{
  "prediction": 0
}
```

- **GET /test/** - Verifica o funcionamento da API:

```json
{
  "message": "API está funcionando corretamente!"
}
```


---

## Contribuindo

1. Faça um fork do repositório.
2. Crie uma nova branch com suas alterações (`git checkout -b feature/nova-feature`).
3. Faça commit das suas alterações (`git commit -m 'Adicionando nova funcionalidade'`).
4. Envie para o seu fork (`git push origin feature/nova-feature`).
5. Abra um Pull Request.

---

## Licença

Este projeto está licenciado sob a [MIT License](LICENSE).

