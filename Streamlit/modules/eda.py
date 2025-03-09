import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

@st.cache_data
def exploratory_data_analysis(df):
    
    # Seção 1.1: Estrutura do Dataset
    with st.expander("📁 **Estrutura do Dataset**", expanded=False):
        st.markdown("""
        ### ❓ **Perguntas-chave**  
        - Quantas observações e variáveis temos?  
        - Como são os primeiros registros?  
        - Quais tipos de dados estamos lidando?  
        - Existem valores ausentes?  
        """)
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown("#### 🔍 **Amostra dos Dados:**")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            st.dataframe(df.head(), height=250)
        with col2:
            st.markdown("#### 📊 **Total de Observações**")
            st.metric("📊 Total de Registros", df.shape[0])
            st.metric("📂 Total de Variáveis", df.shape[1])
        with col3:
            st.markdown("#### 🛠 **Tipos de Dados e Valores Ausentes:**")
            missing_data = pd.DataFrame({
                'Tipo': df.dtypes,
                'Valores Ausentes': df.isnull().sum(),
                '% Ausentes': (df.isnull().mean() * 100).round(2)
            })

            # Exibindo os dados no Streamlit
            st.dataframe(missing_data, hide_index=False)

    
    # Seção 1.2: Distribuição das Notas Fiscais
    with st.expander("🧾 **Distribuição das Notas Fiscais**", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### ❓ **Perguntas-chave**  
            - Como as notas fiscais estão distribuídas entre válidas e inválidas?  
            - Qual a porcentagem de cada classe?  
            """)
        with col2:
            st.markdown("### 📊 **Distribuição das Notas Fiscais**")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.countplot(x='class_label', data=df, order=['valid', 'not valid'], ax=ax)
            ax.set_title('Distribuição de Notas Fiscais Válidas vs. Inválidas', fontsize=14)
            ax.set_xlabel('Classificação')
            ax.set_ylabel('Contagem')
            total = len(df)
            for p in ax.patches:
                percentage = f'{100 * p.get_height()/total:.1f}%'
                x = p.get_x() + p.get_width() / 2
                y = p.get_height() + 10
                ax.annotate(percentage, (x, y), ha='center')
            st.pyplot(fig)
    
    # Seção 1.3: Análise das Variáveis Numéricas
    with st.expander("📊 **Distribuição das Variáveis Numéricas**", expanded=False):
        col1, col2 = st.columns([0.9, 1])
        with col1:
            st.markdown("""
            ### ❓ **Perguntas-chave**  
            - Quais são os valores principais das variáveis numéricas?  
            - Existem outliers nas taxas de impostos?  
            """)
            numeric_cols = [
                'calculated_value', 'iss_tax_rate', 'inss_tax_rate',
                'csll_tax_rate', 'ir_tax_rate', 'cofins_tax_rate', 'pis_tax_rate'
            ]
            st.markdown("### 📌 **Estatísticas Descritivas**")
            st.dataframe(df[numeric_cols].describe().T.style.format("{:.2f}"))
        with col2:
            st.markdown("### 📉 **Outliers nas Taxas de Impostos**")
            fig, ax = plt.subplots(figsize=(12, 6))
            tax_cols = [col for col in numeric_cols if 'tax_rate' in col]
            sns.boxplot(data=df[tax_cols], orient='h', ax=ax)
            ax.set_title('Distribuição das Taxas de Impostos', fontsize=14)
            ax.set_xlabel('Valor (%)')
            st.pyplot(fig)
    
    # Seção 1.4: Distribuição das Variáveis Categóricas
    with st.expander("🏷️ **Distribuição das Variáveis Categóricas**", expanded=False):
        col1, col2 = st.columns([0.9, 1])
        with col1:
            st.markdown("""
            ### ❓ **Perguntas-chave**  
            - Como as notas fiscais estão distribuídas por estado?  
            - Existe alguma relação entre o estado e a validade da nota?  
            """)
        with col2:
            st.markdown("### 📌 **Distribuição das Notas por Estado**")
            fig, ax = plt.subplots(figsize=(12, 6))
            order = df['state'].value_counts().index
            sns.countplot(x='state', hue='class_label', data=df, order=order, ax=ax)
            ax.set_title('Validade das Notas por Estado', fontsize=14)
            ax.set_xlabel('Estado', fontsize=12)
            ax.set_ylabel('Quantidade', fontsize=12)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            ax.legend(title='Classificação das Notas', labels=['Válidas', 'Inválidas'])
            st.pyplot(fig)
    
    
    # Seção 1.6: Matriz de Correlação
    with st.expander("🔗 **Matriz de Correlação entre Variáveis**", expanded=False):
        col1, col2 = st.columns([0.9, 1])
        with col1:
            st.markdown("""
            ### ❓ **Perguntas-chave**  
            - Como as variáveis numéricas estão correlacionadas entre si?  
            """)
        with col2:
            df_corr = df.copy()
            le = LabelEncoder()
            df_corr['class_label'] = le.fit_transform(df_corr['class_label'])
            st.markdown("### 📊 **Matriz de Correlação entre Variáveis Numéricas**")
            fig, ax = plt.subplots(figsize=(14, 10))
            corr_matrix = df_corr[numeric_cols + ['class_label']].corr()
            sns.heatmap(
                corr_matrix, annot=True, cmap='coolwarm', fmt=".2f",
                mask=np.triu(np.ones_like(corr_matrix, dtype=bool)), ax=ax
            )
            ax.set_title('Matriz de Correlação entre Variáveis Numéricas', fontsize=14)
            st.pyplot(fig)