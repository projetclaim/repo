#avant toute utilisation, lancer la commande suivante pour installer les differents modules via l'invite de commande
#pip3 install pandas streamlit numpy seaborn matplotlib scikit_learn

#Ensuite, utiliser l'invite de commande suivant pour ouvrir le streamlit
#streamlit run https://github.com/projetclaim/repo/blob/main/streamlit.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

header = st.beta_container()
dataset = st.beta_container()
features = st.beta_container()
modelTraining = st.beta_container()

st.title('Projet Claim Generali')
st.image('./generalilogo.png')

st.markdown('Texte1:')
st.markdown('Texte2 ')

st.sidebar.title('Réglages du modèle Regression Logistique')
st.sidebar.header('Solver')
solver = st.sidebar.selectbox(label='Solver', options=['newton-cg', 'lbfgs', 'liblinear'])
st.sidebar.header('Hyperparamètres')
C = st.sidebar.slider(label='C', min_value=.1, max_value=1.1, step=.1)  
