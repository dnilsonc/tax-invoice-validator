import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import io
from sklearn.compose import ColumnTransformer

 
# Definição das colunas
tax_cols = [
    'iss_tax_rate', 'inss_tax_rate', 'csll_tax_rate',
    'ir_tax_rate', 'cofins_tax_rate', 'pis_tax_rate'
]
categorical_features = ['state', 'lc116']
binary_features = ['iss_retention', 'opting_for_simples_nacional']
numerical_features = tax_cols + ['tax_alert']

# Definição do preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', 'passthrough', categorical_features),
        ('binary', 'passthrough', binary_features)
    ],
    remainder='passthrough'
)


@st.cache_data
def data_preprocessing(df):
    
    # Seção 2.1: Tratamento de Valores Ausentes e Outliers
    with st.expander("📌 **Tratamento de Valores Ausentes e Outliers**", expanded=False):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("""
            ### ❓ **Perguntas-chave**  
            - **Quais colunas podem ser removidas por serem irrelevantes?**  
            - **Qual a estratégia de imputação para valores ausentes?**  
            - **Como tratar outliers nas variáveis numéricas?**  
            """)
            st.markdown("### 🧹 **Limpeza Inicial e Remoção de Colunas Irrelevantes**")
            st.markdown("""
            Para melhorar a modelagem, removemos colunas desnecessárias, como:
            - **id**
            - **issue_date**
            - **id_supplier**
            """)
            df_clean = df.drop(['id', 'issue_date', 'id_supplier'], axis=1)
            st.markdown("### 🏗️ **Tratamento de Valores Ausentes**")
            missing_strategy = {
                'iss_tax_rate': df['iss_tax_rate'].median(),
                'state': 'Desconhecido',
                'calculated_value': df['calculated_value'].median()
            }
            df_clean = df_clean.fillna(missing_strategy)
            st.write("- **Dados após tratamento de valores ausentes:**")
            st.dataframe(df_clean.head(), height=250)
        with col2:
            st.markdown("### 📉 **Tratamento de Outliers**")
            st.markdown("""
            Aplicamos np.clip() para limitar valores extremos e evitar impacto negativo no modelo.
            """)
            df_clean['calculated_value'] = np.clip(
                df_clean['calculated_value'],
                df_clean['calculated_value'].quantile(0.01),
                df_clean['calculated_value'].quantile(0.99)
            )
            for col in tax_cols:
                df_clean[col] = np.clip(df_clean[col], 0, 100)
            st.write("- **Dados após tratamento de outliers:**")
            st.dataframe(df_clean.head(), height=250)
            st.markdown("### 🛠️ **Informações do DataFrame Limpo**")
            buffer = io.StringIO()
            df_clean.info(buf=buffer)
            info_str = buffer.getvalue()
            info_lines = info_str.split("\n")[5:-2]
            info_data = []
            for line in info_lines:
                parts = line.split()
                if len(parts) >= 4:
                    col_name = " ".join(parts[1:-2])
                    info_data.append([col_name, parts[-2], parts[-1]])
            info_df = pd.DataFrame(info_data, columns=["Coluna", "Valores Não Nulos", "Tipo"])
            st.dataframe(info_df)
    
    # Seção 2.2: Engenharia de Features
    with st.expander("🛠️ **Engenharia de Features**", expanded=False):
        col1, col2 = st.columns([0.8, 1])
        with col1:
            st.markdown("""
            ### ❓ **Perguntas-chave**  
            - **Quais novas features podemos criar para melhorar a performance do modelo?**  
            - **Como utilizar insights da correlação para criar variáveis?**  
            """)
            st.markdown("### 🚨 **Criação da Feature tax_alert**")
            st.markdown("""
            Baseado na análise de correlação, criamos um alerta (tax_alert) para taxas de impostos suspeitas:
            - csll_tax_rate > 1.5
            - pis_tax_rate > 2.5
            """)
            df_clean['tax_alert'] = (
                (df_clean['csll_tax_rate'] > 1.5) |
                (df_clean['pis_tax_rate'] > 2.5)
            ).astype(int)
        with col2:
            st.write("- **Exemplo de dados com a nova feature tax_alert**:")
            st.dataframe(df_clean.head(), height=250)
    
    # Seção 2.3: Codificação de Variáveis Categóricas
    with st.expander("🔄 **Codificação de Variáveis Categóricas**", expanded=False):
        st.markdown("""
        ### ❓ **Perguntas-chave**  
        - **Quais variáveis precisam de codificação para uso em modelos de Machine Learning?**  
        - **Quais técnicas de codificação serão aplicadas?**  
        """)
        st.markdown("### 🏷️ **Variáveis que Precisam de Codificação**")
        categorical_features = ['state', 'lc116']
        binary_features = ['iss_retention', 'opting_for_simples_nacional']
        numerical_features = tax_cols + ['tax_alert']
        encoding_df = pd.DataFrame({
            "Coluna": categorical_features + binary_features,
            "Tipo": ["Categórica"] * len(categorical_features) + ["Binária"] * len(binary_features),
        })
        st.dataframe(encoding_df, hide_index=True)
        st.markdown("### 🔠 **Aplicando LabelEncoder e StandardScaler**")
        label_encoder = LabelEncoder()
        for col in categorical_features:
            df_clean[col] = label_encoder.fit_transform(df_clean[col])

        # Mapear a variável alvo
        df_clean['class_label'] = df_clean['class_label'].map({'valid': 0, 'not valid': 1})
        
        st.write("- **Dados preparados para o *pipeline* de pré-processamento.**")
        st.dataframe(df_clean.head(), height=250)
        st.markdown("---")
    return df_clean



