# Usar a imagem base do Python
FROM python:3.10

# Definir o diretório de trabalho dentro do contêiner
WORKDIR /app

# Copiar todos os arquivos do diretório atual para o contêiner
COPY . .

# Instalar as dependências do projeto
RUN pip install --no-cache-dir -r Streamlit/requirements.txt
RUN pip install --no-cache-dir -r API/requirements.txt

# Expor as portas onde o Streamlit e a API estarão escutando
EXPOSE 8000
EXPOSE 8501

# Comando para rodar o Streamlit e a API utilizando uvicorn
CMD ["sh", "-c", "uvicorn API.api:app --host 0.0.0.0 --port 8000 & streamlit run Streamlit/app.py --server.port 8501"]
