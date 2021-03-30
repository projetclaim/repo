# Repo-du-projet
Repo du projet
# Projet Claim
#Test flo
#Test flo2
#test flo3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline 
import seaborn as sns
sns.set_theme()
X_train= pd.read_csv('X_train.csv', index_col= 0)
X_test= pd.read_csv('X_test.csv')
y_train= pd.read_csv('y_train.csv', index_col= 0)

df= X_train.merge(right= y_train, on= ['Identifiant'])
df= df.drop(['Identifiant'], axis= 1)
df.head()

#            Découpage des variables; superficief, ft_22_categ et Insee

*Découpage de la variable superficief en 4 classes: 'petit', 'moyen', 'moyen+' et 'grand'

df['superf']=pd.cut(df['superficief'],[0,7686,15372,23058,30745],labels=['petit','moyen','moyen+','grand'])

*Découpage de la variable ft_22_categ par décennie en 7 classes:  
'avant 1960s', '1970s', '1980s', '1990s', '2000s', '2010s', 'jusqu'à 2016'.

df['date']=pd.cut(df['ft_22_categ'],[0,1960,1970,1980,1990,2000,2010,2016],
                  labels=['avant 1960s','1970s','1980s','1990s','2000s','2010s',"jusqu'à 2016"])

*Création de 2 nouvelles variables à partir de la variable Insee.
           Dep: correspond aux codes des départements, regroupement avec les 2 premiers chiffres de la variable Insee.
           Ville: correspond aux codes des villes, regroupement avec les 3 derniers chiffres de la variable Insee.
      

df['dep']=df['Insee'].str[:2]

df['ville']=df['Insee'].str[3:]

#                                               DataViz

cor = df.corr()
plt.figure(figsize=(12,12))

sns.heatmap(cor, annot= True, cmap="coolwarm")
plt.title('Heatmap', fontsize= 20);

Cette "heatmap" nous permet de mesurer les relations entre chaque paire de variables qualitatives.
On remarque des corrélations entre les variables: ft_4 et ft_21, ft_19 et ft_22, ft_22 et superficief, ft_19 et superficief, target et superficie, target et ft_21


                          ------------------------------------------------------------




Le graphe ci-dessus "La linéarité entre la superficie et la target" nous permet de vérifier l'hypothèse de linéarité entre la superficie et la variable cible.
Plus la superficie est grande plus le risque d'avoir un sinistre augmente.

                          ------------------------------------------------------------

df_b=df.ft_24_categ.value_counts().sort_index()
sns.barplot(y='target',x='ft_24_categ',data=df,order=df_b.index)
plt.title('Distribution de la variable ft_24_categ en fonction de target');


Le graphe ci-dessus "Distribution de ft_24_categ en fonction de target" nous permet de vérifier la dépendance entre la variable ft_24_categ et la variable cible.
Plus la valeur de ft_24_categ est grande plus le risque est élevé.

                           ------------------------------------------------------------

sns.barplot(x='date',y='target',data=df)
plt.title('Distribution de la variable date en fonction de target');

Le graphe ci-dessus "Distribution de la variable date en fonction de target" nous permet de vérifier l'influence de l'année de construction sur la variable cible.  

                          ------------------------------------------------------------

sns.barplot(x='ft_7_categ',y='target',data=df)
plt.title('Distribution de la variable ft_7_categ en fonction de target');

Le graphe ci-dessus "Distribution de la variable ft_7_categ en fonction de target" nous permet de vérifier l'influence de cette variable sur la variable cible.
On remarque que ft_7_categ= 2 représente un risque plus élevé pour avoir un sinistre.
