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

st.sidebar.title('Setting')
st.sidebar.header('x settings')
x_distribution = st.sidebar.selectbox(label='Choose X distribution', options=['uniform', 'normal', 'beta'])
nb_of_lines = st.sidebar.slider(label='Number of observations', min_value=100, max_value=10000)
st.sidebar.header('y settings')
sigma = st.sidebar.slider(label='sigma', min_value=0., max_value=2., value=.5)
beta1 = st.sidebar.slider(label='beta1', min_value=-3., max_value=3., value=.0)
beta2 = st.sidebar.slider(label='beta2', min_value=-3., max_value=3., value=.0)
beta3 = st.sidebar.slider(label='beta3', min_value=-3., max_value=3., value=.0)
alpha = st.sidebar.slider(label='alpha', min_value=-3., max_value=3., value=.0)    

