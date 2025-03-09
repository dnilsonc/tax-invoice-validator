import pandas as pd
import streamlit as st

@st.cache_data
def load_data():
    url = 'https://drive.google.com/uc?export=download&id=1Lz64hDUd_EKHnmx5f6ONIYqOp_cYZkvp'
    return pd.read_csv(url, sep=';')

