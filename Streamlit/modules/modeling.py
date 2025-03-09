import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif
from modules.preprocessing import preprocessor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# -----------------------------------------------------------------------------
# ⚙️ Funções Auxiliares para Treino e Avaliação
# -----------------------------------------------------------------------------

def train_model(model, X_train, y_train, X_val, y_val):
    """Treina e avalia um modelo com exibição aprimorada no Streamlit."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    
    st.metric("🎯 Acurácia", f"{accuracy:.4f}")
    return model

def evaluate_model(model, X_val, y_val, model_name):
    """Avaliação do desempenho do modelo com métricas formatadas e gráficos lado a lado."""
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    # Formatando o relatório de classificação em DataFrame
    report_dict = classification_report(y_val, y_pred, target_names=['valid', 'not valid'], output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose().round(3)

    # Criando layout com 3 colunas para melhor visualização
    col1, col2, col3 = st.columns([0.8, 0.95, 1])

    with col1:
        st.markdown("### 📌 **Métricas de Classificação**")
        st.dataframe(report_df.style.format("{:.3f}"))

    with col2:
        # Matriz de Confusão
        st.markdown("### 📌 **Matriz de Confusão**")
        fig, ax = plt.subplots()  # Ajusta o tamanho da figura
        cm = confusion_matrix(y_val, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Válido', 'Inválido'], yticklabels=['Válido', 'Inválido'])
        ax.set_xlabel("Predito")
        ax.set_ylabel("Real")
        ax.set_title(f'Matriz de Confusão')

        # Ajustar layout para ocupar o máximo de espaço possível
        fig.tight_layout()
        st.pyplot(fig)

    with col3:
        st.markdown("### 📌 **Curva ROC**")
        fig, ax = plt.subplots()  # Ajusta o tamanho da figura
        fpr, tpr, _ = roc_curve(y_val, y_proba)
        auc_score = roc_auc_score(y_val, y_proba)
        ax.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}', linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
        ax.set_xlabel('Falsos Positivos')
        ax.set_ylabel('Verdadeiros Positivos')
        ax.set_title('Curva ROC')
        ax.legend()

        # Ajustar layout para ocupar o máximo de espaço possível
        fig.tight_layout()
        st.pyplot(fig)

@st.cache_data
def model_training_and_evaluation(df_clean):
    # -----------------------------------------------------------------------------
    # 📌 Seção 3.1: Divisão de Dados
    # -----------------------------------------------------------------------------
    with st.expander("📌 **Divisão de Dados**", expanded=False):
        st.markdown("""
        ### ❓ **Perguntas-chave**  
        - **Como dividir o dataset em treinamento, validação e teste?**  
        - **Por que usar divisão estratificada?**  
        """)

        # Separação em features (X) e alvo (y)
        X = df_clean.drop(columns=['class_label'])
        y = df_clean['class_label']

        # Aplicando Pipeline de Pré-processamento
        X = pd.DataFrame(preprocessor.fit_transform(X), columns=X.columns)

        # Divisão dos dados
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, stratify=y, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, stratify=y_train, random_state=42)

        # Exibir resumo da divisão dos dados
        st.markdown("### 📊 **Resumo da Divisão dos Dados:**")
        st.write(f"- **Treinamento:** {len(X_train)} amostras ({len(X_train)/len(df_clean)*100:.2f}%)")
        st.write(f"- **Validação:** {len(X_val)} amostras ({len(X_val)/len(df_clean)*100:.2f}%)")
        st.write(f"- **Teste:** {len(X_test)} amostras ({len(X_test)/len(df_clean)*100:.2f}%)")

    # -----------------------------------------------------------------------------
    # 📌 Seção 3.2: Treinamento e Avaliação dos Modelos
    # -----------------------------------------------------------------------------
    with st.expander("📌 **Treinamento e Avaliação dos Modelos**", expanded=False):
        st.markdown("""
        ### ❓ **Perguntas-chave**  
        - **Quais modelos clássicos podemos testar?**  
        - **Como avaliar o desempenho com métricas como acurácia e matriz de confusão?**  
        """)

                # Criando abas para os modelos
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["▫️ KNN", "▫️ SVM", "▫️ Naive Bayes", "▫️ XGBoost", "▫️ Árvore de Decisão", "▫️ Random Forest"])

        # Modelo 1: KNN
        with tab1:
            knn_model = KNeighborsClassifier()
            knn_model = train_model(knn_model, X_train, y_train, X_val, y_val)
            evaluate_model(knn_model, X_val, y_val, "KNN")
        
        # Modelo 2: SVM
        with tab2:
            svm_model = SVC(probability=True, random_state=42)
            svm_model = train_model(svm_model, X_train, y_train, X_val, y_val)
            evaluate_model(svm_model, X_val, y_val, "SVM")

        # Modelo 3: Naive Bayes
        with tab3:
            nb_model = GaussianNB()
            nb_model = train_model(nb_model, X_train, y_train, X_val, y_val)
            evaluate_model(nb_model, X_val, y_val, "Naive Bayes")

        # Modelo 4: XGBoost
        with tab4:
            bst = XGBClassifier(random_state=42, n_jobs=-1)
            bst = train_model(bst, X_train, y_train, X_val, y_val)
            evaluate_model(bst, X_val, y_val, "XGBoost")

        # Modelo 5: Árvore de Decisão
        with tab5:
            dt_model = DecisionTreeClassifier(random_state=42)
            dt_model = train_model(dt_model, X_train, y_train, X_val, y_val)
            evaluate_model(dt_model, X_val, y_val, "Árvore de Decisão")

        # Modelo 6: Random Forest
        with tab6:
            rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
            rf_model = train_model(rf_model, X_train, y_train, X_val, y_val)
            evaluate_model(rf_model, X_val, y_val, "Random Forest")

    # =============================================================================
    # 🏋️ Seção 3.3: Balanceamento de Dados e Re-treinamento dos Modelos
    # =============================================================================
    with st.expander("📌 **Balanceamento de Dados e Re-treinamento**", expanded=False):
        st.markdown("""
        ### ❓ **Perguntas-chave**  
        - **O balanceamento da base melhora o desempenho dos modelos?**  
        - **Como os modelos se comportam após a remoção do viés de classes?**  
        """)

        # -------------------------------------------------------------------------
        # 🎯 Aplicação do Balanceamento com RandomUnderSampler
        # -------------------------------------------------------------------------
        st.subheader("📊 **Balanceamento de Dados com RandomUnderSampler**")

        # Inicializar o RandomUnderSampler
        undersampler = RandomUnderSampler(random_state=42)

        # Aplicar o undersampling nos dados
        X_resampled, y_resampled = undersampler.fit_resample(X, y)

        # Encontrar as amostras removidas
        X_discarded = X[~X.index.isin(X_resampled.index)]
        y_discarded = y[~y.index.isin(y_resampled.index)]
        discarded_data = pd.concat([X_discarded, y_discarded], axis=1)

        # Aplicar o undersampling
        X, y = X_resampled, y_resampled

        # Exibir informações sobre a redução da base
        col1, col2, col3 = st.columns(3)
        col1.metric("📉 Tamanho Antes", len(df_clean))
        col2.metric("📊 Tamanho Após", len(X))
        col3.metric("🚫 Amostras Removidas", len(discarded_data))

        # -------------------------------------------------------------------------
        # 📌 Divisão de Dados Após Balanceamento
        # -------------------------------------------------------------------------
        st.subheader("📊 **Divisão de Dados Após Balanceamento**")

        # Divisão estratificada
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.10, stratify=y, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.10, stratify=y_temp, random_state=42)

        # Exibir resumo das divisões
        col1, col2, col3 = st.columns(3)
        col1.metric("▫️ Treino", f"{len(X_train)} ({(len(X_train) / len(X) * 100):.2f}%)")
        col2.metric("▫️ Validação", f"{len(X_val)} ({(len(X_val) / len(X) * 100):.2f}%)")
        col3.metric("▫️ Teste", f"{len(X_test)} ({(len(X_test) / len(X) * 100):.2f}%)")

        # -------------------------------------------------------------------------
        # 📌 Re-treinamento dos Modelos
        # -------------------------------------------------------------------------
        st.subheader("📊 **Re-treinamento dos Modelos**")

        # Criando abas para os modelos
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["▫️ SVM", "▫️ KNN",  "▫️ Naive Bayes", "▫️ XGBoost", "▫️ Árvore de Decisão", "▫️ Random Forest"])

        # Modelo 1: SVM
        with tab1:
            svm_model = SVC(probability=True, random_state=42)
            svm_model = train_model(svm_model, X_train, y_train, X_val, y_val)
            evaluate_model(svm_model, X_val, y_val, "SVM")

        # Modelo 2: KNN
        with tab2:
            knn_model = KNeighborsClassifier(n_jobs=-1)
            knn_model = train_model(knn_model, X_train, y_train, X_val, y_val)
            evaluate_model(knn_model, X_val, y_val, "KNN")

        # Modelo 3: Naive Bayes
        with tab3:
            nb_model = GaussianNB()
            nb_model = train_model(nb_model, X_train, y_train, X_val, y_val)
            evaluate_model(nb_model, X_val, y_val, "Naive Bayes")

        # Modelo 4: XGBoost
        with tab4:
            bst = XGBClassifier(random_state=42, n_jobs=-1)
            bst = train_model(bst, X_train, y_train, X_val, y_val)
            evaluate_model(bst, X_val, y_val, "XGBoost")

        # Modelo 5: Árvore de Decisão
        with tab5:
            dt_model = DecisionTreeClassifier(random_state=42)
            dt_model = train_model(dt_model, X_train, y_train, X_val, y_val)
            evaluate_model(dt_model, X_val, y_val, "Árvore de Decisão")

        # Modelo 6: Random Forest
        with tab6:
            rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
            rf_model = train_model(rf_model, X_train, y_train, X_val, y_val)
            evaluate_model(rf_model, X_val, y_val, "Random Forest")


    # =============================================================================
    # 🔍 Seção 3.4: Importância das Features
    # =============================================================================
    with st.expander("📌 **Importância das Features**", expanded=False):
        st.markdown("""
        ### ❓ **Perguntas-chave**  
        - **Quais variáveis mais influenciam a decisão dos modelos?**  
        - **Como podemos quantificar a relevância de cada variável?**  
        """)

        # Criando colunas para dispor os gráficos lado a lado
        col1, col2 = st.columns(2)

        feature_names = X.columns
        rf_importances = rf_model.feature_importances_
        sorted_idx = np.argsort(rf_importances)[-10:]  # Top 10 features

        df_corr = pd.concat([X, y], axis=1)
        correlation_matrix = df_corr.corr()
        corr_with_target = correlation_matrix['class_label'].sort_values(ascending=False)[1:]

        X_categoric = df_corr.drop('class_label', axis=1)
        mi = mutual_info_classif(X_categoric, df_corr['class_label'])
        mi_scores = pd.Series(mi, index=X_categoric.columns).sort_values(ascending=False)

        X_numeric = df_corr.drop(columns=['class_label'])
        y_numeric = df_corr['class_label']
        selector = SelectKBest(f_classif, k='all')
        selector.fit(X_numeric, y_numeric)
        anova_scores = pd.Series(selector.scores_, index=X_numeric.columns).sort_values(ascending=False)

        # -------------------------------------------------------------------------
        # 🔍 Feature Importance - Random Forest (Gráfico 1)
        # -------------------------------------------------------------------------
        with col1:
            st.markdown("### 📌 **Importância das Features - Random Forest**")
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.barh(range(len(sorted_idx)), rf_importances[sorted_idx], align='center', color='skyblue')
            ax.set_yticks(range(len(sorted_idx)))
            ax.set_yticklabels([feature_names[i] for i in sorted_idx])
            ax.set_title("Top 10 Features Mais Importantes - Random Forest")
            ax.set_xlabel('Importância', fontsize=12)
            st.pyplot(fig)

        # -------------------------------------------------------------------------
        #  ANOVA F-score (Gráfico 2)
        # -------------------------------------------------------------------------
        with col2:
            st.markdown("### 📌 **ANOVA F-Score**")
            fig, ax = plt.subplots(figsize=(7, 5))
            x = np.arange(len(anova_scores))
            y = anova_scores.values
            ax.barh(x, y, color='lightcoral')
            ax.set_yticks(x)
            ax.set_yticklabels(anova_scores.index)
            ax.set_title("ANOVA F-Score das Variáveis Numéricas", fontsize=14)
            ax.set_xlabel("F-Score")
            st.pyplot(fig)

        # Criando a segunda linha de colunas
        col3, col4 = st.columns(2)

        # -------------------------------------------------------------------------
        # 🧠 Informação Mútua para Variáveis Categóricas (Gráfico 3)
        # -------------------------------------------------------------------------
        with col3:
            st.markdown("### 📌 **Informação Mútua para Variáveis Categóricas**")
            fig, ax = plt.subplots(figsize=(10, 5))
            x = np.arange(len(mi_scores))
            y = mi_scores.values
            ax.bar(x, y, color='mediumseagreen')
            ax.set_title("Informação Mútua com 'class_label'", fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels(mi_scores.index, rotation=45, ha='right')
            ax.set_ylabel("Informação Mútua")
            st.pyplot(fig)

        # -------------------------------------------------------------------------
        # 📈 📊 Correlação de Pearson (Gráfico 4)
        # -------------------------------------------------------------------------
        with col4:
            st.markdown("### 📌 **Correlação de Pearson**")
            fig, ax = plt.subplots(figsize=(10, 5.1))
            x = np.arange(len(corr_with_target))
            y = corr_with_target.values
            ax.bar(x, y, color='goldenrod')
            ax.set_title("Correlação de Pearson com 'class_label'", fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels(corr_with_target.index, rotation=45, ha='right')
            ax.set_ylabel("Correlação")
            st.pyplot(fig)
    # =============================================================================
    # 🎯 Seção 3.6: Redução de Features e Re-treinamento
    # =============================================================================
    with st.expander("📌 **Redução de Features e Re-treinamento**", expanded=False):
        st.markdown("""
        ### ❓ **Pergunta-chave**  
        - **Quais benefícios podemos obter ao reduzir o número de variáveis?**  
        """)

        st.subheader("📊 **Seleção das Top 5 Features**")
        
        # Seleção das Top 5 Features
        selected_features = list(mi_scores[:5].index)
        
        # Criar DataFrame formatado
        top_features_df = pd.DataFrame({"Ranking": range(1, 6), "Feature": selected_features})

        # Exibição dos dados sem índice
        st.dataframe(top_features_df, hide_index=True)

        # Filtrando o conjunto de treino e validação com as features selecionadas
        X_train_filtered = X_train[selected_features]
        X_val_filtered = X_val[selected_features]
        X_test_filtered = X_test[selected_features]  # Não se esqueça de filtrar o conjunto de teste também!

        # Criando abas para os modelos
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["▫️ Naive Bayes", "▫️ SVM", "▫️ KNN", "▫️ Árvore de Decisão", "▫️ Random Forest", "▫️ XGBoost"])

        # Modelo 1: Naive Bayes
        with tab1:
            nb_model = GaussianNB()
            nb_model = train_model(nb_model, X_train_filtered, y_train, X_val_filtered, y_val)
            evaluate_model(nb_model, X_val_filtered, y_val, "Naive Bayes")

        # Modelo 2: SVM
        with tab2:
            svm_model = SVC(probability=True, random_state=42)
            svm_model = train_model(svm_model, X_train_filtered, y_train, X_val_filtered, y_val)
            evaluate_model(svm_model, X_val_filtered, y_val, "SVM")
        
        # Modelo 3: KNN
        with tab3:
            knn_model = KNeighborsClassifier(n_jobs=-1)
            knn_model = train_model(knn_model, X_train_filtered, y_train, X_val_filtered, y_val)
            evaluate_model(knn_model, X_val_filtered, y_val, "KNN")

        # Modelo 4: Árvore de Decisão
        with tab4:
            dt_model = DecisionTreeClassifier(random_state=42)
            dt_model = train_model(dt_model, X_train_filtered, y_train, X_val_filtered, y_val)
            evaluate_model(dt_model, X_val_filtered, y_val, "Árvore de Decisão")

        # Modelo 5: Random Forest
        with tab5:
            rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
            rf_model = train_model(rf_model, X_train_filtered, y_train, X_val_filtered, y_val)
            evaluate_model(rf_model, X_val_filtered, y_val, "Random Forest")

        # Modelo 6: XGBoost
        with tab6:
            bst = XGBClassifier(random_state=42, n_jobs=-1)
            bst = train_model(bst, X_train_filtered, y_train, X_val_filtered, y_val)
            evaluate_model(bst, X_val_filtered, y_val, "XGBoost")


    # =============================================================================
    # 📊 Seção 3.7: Avaliação no Conjunto de Teste
    # =============================================================================
    with st.expander("📌 **Avaliação no Conjunto de Teste**", expanded=False):
        st.markdown("""
        ### ❓ **Pergunta-chave**  
        - **Qual a performance final do melhor modelo no conjunto de teste independente?**  
        """)

        st.subheader("▫️ **Random Forest**")
        # Escolhendo o melhor modelo (exemplo: Random Forest)
        best_model = rf_model  
        X_test_filtered = X_test[selected_features]

        # Avaliação no conjunto de teste
        evaluate_model(best_model, X_test_filtered, y_test, "Random Forest")

    return rf_model, preprocessor 
