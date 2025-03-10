# Projeto de Detecção de Notas Fiscais com Impostos Incorretos 💡

Este projeto tem como objetivo detectar notas fiscais com impostos incorretos utilizando técnicas de Machine Learning. A solução é composta por duas partes principais:

1. **API** (com FastAPI) para disponibilizar o modelo de forma escalável.  
2. **[Aplicação Streamlit](http://34.229.94.169:8501/)** para análise de dados, visualização e testes interativos.

<img src="\img\app.png" style="height:25rem;" />

## Google Colab ☁️
Você pode acessar o notebook do projeto no [Google Colab](https://colab.research.google.com/drive/1dnBPHeDEy3yflbQbv-ZfqcXRM2wXbTZA?usp=sharing).

---

## Sumário 📚
- [Descrição do Projeto](#descrição-do-projeto)
- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Contribuindo](#contribuindo)
- [Licença](#licença)

---

## Descrição do Projeto 📊

O foco do projeto é identificar notas fiscais com valores de impostos inconsistentes. O modelo de classificação (por exemplo, Random Forest) foi treinado para distinguir notas válidas de notas inválidas, considerando diversas variáveis como ISS, INSS, CSLL, PIS, COFINS, entre outras.

### Principais Etapas 🚀
1. **Análise Exploratória (EDA):** Explorar e entender o conjunto de dados.
2. **Pré-processamento:** Limpeza, tratamento de dados ausentes e engenharia de atributos.
3. **Modelagem:** Treinamento do modelo de classificação, avaliação e definição de métricas.
4. **API:** Disponibilização do modelo via FastAPI para previsão.
5. **Interface Streamlit:** Visualização dos resultados e interação com o modelo.

---

## Tecnologias Utilizadas 🛠️

- **Python 3.9+**
- **FastAPI:** API de previsão.
- **Streamlit:** Interface web interativa.
- **Pandas** & **Numpy:** Manipulação de dados.
- **Scikit-learn** & **XGBoost:** Modelagem e Machine Learning.
- **Joblib:** Serialização do modelo.
- **Docker:** Conteinerização (opcional).

---

## Contribuindo 🤝

1. Faça um fork do repositório.
2. Crie uma nova branch para suas alterações (`git checkout -b feature/nova-feature`).
3. Realize commit das suas mudanças (`git commit -m 'Adicionando nova funcionalidade'`).
4. Envie sua branch para o fork (`git push origin feature/nova-feature`).
5. Abra um Pull Request para revisão.

---

## Licença 📜

Este projeto está licenciado sob a [MIT License](LICENSE).
