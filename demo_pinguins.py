import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


#st.image("https://github.com/projetclaim/repo/blob/main/penguins.png", width=550)
st.title("Démo Streamlit : Pinguins")

st.markdown("Objectif : Prédire la classe d'un pinguins")

st.markdown('Chargement des données')
df = pd.read_csv('penguins_size.csv')

if st.checkbox('Afficher les données'):
   df

st.sidebar.header('Variables du jeu de données')
st.sidebar.write(df.columns)

st.markdown('Affichons les valeurs manquantes')

st.write(df.isna().sum())

st.markdown('Remplaçons les valeurs manquantes')
with st.echo():
    df.sex.fillna(df.sex.mode()[0], inplace=True)
    df.fillna(df.mean(), inplace=True)


st.markdown('Afficher la distribution de la var *body_mass_g*')
fig = sns.displot(df.body_mass_g, kde=True)
st.pyplot(fig)


st.markdown('Features à afficher')
xy = st.multiselect('Selectionne une feature en abscisse et une ordonnées',
                            ['culmen_length_mm','culmen_depth_mm','flipper_length_mm','body_mass_g'],
                            ['culmen_length_mm','culmen_depth_mm'])


# Afficher la relation entre 2 variables, et colorer les points en fonctions de l'espèce
fig = plt.figure()
sns.scatterplot(data=df, x=xy[0],
                y=xy[1], hue='species')
plt.title(str(xy[0]) +' vs ' + str(xy[1]));
st.pyplot(fig)


data = df.drop('species', axis =1)
target = df.species
data = pd.get_dummies(data)

trainsize = st.sidebar.slider(label = "Choix de la taille de l‘échantillon d'entrainement",
    min_value = 0.2, max_value = 1.0,step = 0.05)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=trainsize)

scaler = st.sidebar.radio(
    "Choix du scaler",
    ('StandardScaler', 'MinMaxScaler'))

if scaler == 'StandardScaler':
    sc = StandardScaler()
else :
    sc = MinMaxScaler()

st.markdown('Scaling des données')
with st.echo():
    X_train_scaled = pd.DataFrame(
        sc.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(sc.transform(X_test), columns=X_train.columns)

st.header("Sélection du modèle & affichage des résultats")
modele_choisi = st.selectbox(label = "Sélection du modèle",
    options = ['KNN','LR','DTC','RandomForest'])

if modele_choisi == 'LR':
    model = LogisticRegression()
elif modele_choisi == 'KNN':
    model = KNeighborsClassifier()
elif modele_choisi == 'DTC':
    model = DecisionTreeClassifier()
elif modele_choisi == 'RandomForest':
    model = RandomForestClassifier()


prediction = model.fit(X_train_scaled, y_train)
score = model.score(X_test_scaled, y_test)
y_pred = prediction.predict(X_test_scaled)
elements = classification_report(y_test, y_pred, output_dict = True)
rapport = pd.DataFrame.from_dict(elements)


if score > 0.80:
    st.success('Le score est de {}'.format(score))
    st.balloons()
else:
    st.error('Le score est de {}'.format(score))

if st.checkbox('Afficher le detail du rapport sur le score'):
   rapport
