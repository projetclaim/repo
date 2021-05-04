#invite de commande
#streamlit run https://github.com/projetclaim/repo/blob/main/streamlit.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

header = st.beta_container()
dataset = st.beta_container()
features = st.beta_container()
modelTraining = st.beta_container()

with header:
    st.title('Projet Claim Generali')
    
st.markdown('Texte:')
st.markdown('Texte2 ')

st.sidebar.title('Réglages du modèle Regression Logistique')
st.sidebar.header('Solver')
solver = st.sidebar.selectbox(label='Solver', options=['newton-cg', 'lbfgs', 'liblinear'])
st.sidebar.header('Hyperparamètres')
C = st.sidebar.slider(label='C', min_value=0.1, max_value=1, step=.1)  

