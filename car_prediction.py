
import os
import numpy as np
import pandas as pd

# encoder
from sklearn.preprocessing import LabelEncoder

# splitting the data
from sklearn.model_selection import train_test_split

# data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# model
from sklearn.ensemble import RandomForestRegressor

# evaluation metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

# fine tuning
from sklearn.model_selection import GridSearchCV

import joblib

# warnings
import warnings
warnings.filterwarnings('ignore')
import pandas as pd #utilisé pour manipuler et analyser des données tabulaires en Python.
from google.colab import files #permet de télécharger et charger des fichiers depuis l'environnement Google Colab.
import io #offre des fonctionnalités d'entrée/sortie pour gérer les flux de données.
import seaborn as sns #est utilisé pour créer des visualisations statistiques attrayantes.
import matplotlib.pyplot as plt #permet de créer des graphiques et des visualisations personnalisées dans Python.
import numpy as np

uploaded = files.upload()

data = pd.read_csv(io.BytesIO(uploaded['Automobile_Tn_F-1 (4).csv']))

df=data.copy()
df.head()

df.tail()

df.shape

df.dtypes

df.isna().sum()

df['Car Brand'].value_counts()

ax = sns.countplot(data=df, x=df['Car Brand'])
ax.tick_params(axis='x', rotation=90)

ax = sns.countplot(data=df, x=df['year'])
ax.tick_params(axis='x', rotation=90)

df['Energie'].value_counts()

labels = ['Manuelle', 'Automatique']
plt.pie(df['gear'].value_counts(), labels = labels, autopct='%.0f%%')
plt.legend()
plt.show()

# distribution of cars by fuel type
labels = ['Essence', 'Diesel', 'Hybride (essence)', 'Hybride (diesel)', 'Hybride rechargeable']
plt.pie(df['Energie'].value_counts(), labels = labels, explode=[0, 0, 1, 1, 1], autopct='%.0f%%')
plt.legend()
plt.show()

vis_1=pd.pivot_table(df, index=['year'],values = ['prix DT'],aggfunc = 'mean')
vis_1.plot(kind='line',linewidth=4.5,figsize=(12,7),title='Average car price by Year')

vis_2=pd.pivot_table(df, index=['Kms'],values = ['prix DT'],aggfunc = 'mean')
vis_2.plot(kind='line',linewidth=4.5,figsize=(12,7),title='Average car price by kilometers')

plt.figure(figsize=(10,7))
sns.heatmap(df[["prix DT","Car Brand","year","Kms","Energie"]].corr(), annot=True,linewidths=.5,fmt='.2f')
plt.title("Correlation Graph",size=18)

sns.pairplot(df)

labelencoder = LabelEncoder()

df.info()

# encoding the car's Brand with label encoder
df['Car Brand'] = labelencoder.fit_transform(df['Car Brand'])

df['Car Model'] = labelencoder.fit_transform(df['Car Model'])

df['Energie'] = labelencoder.fit_transform(df['Energie'])

df['gear'] = labelencoder.fit_transform(df['gear'])

df.head()

X = df.drop('prix DT', axis=1)
y = df['prix DT']

X.head(), X.shape

y.head(), y.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# train
regr = RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42)
regr.fit(X_train, y_train.values.ravel())

predictions = regr.predict(X_test)

plt.scatter(predictions,y_test)
plt.title('Prediction and Original data correlation')

regr.score(X_train, y_train)

mse = mean_squared_error(y_test.values.ravel(), predictions)
mae = mean_absolute_error(y_test.values.ravel(), predictions)
r2 = r2_score(y_test.values.ravel(), predictions)

# results
print(f"MSE: {round(mse, 2)}")
print(f"MAE: {round(mae, 2)}")
print(f"R2 Score: {round(r2, 2)}")

joblib.dump(labelencoder,'label_encoder.pkl')

joblib.dump(regr,'carPrediction.pkl')

"""SEARCH GRID"""

parameters = {
    'max_depth': [10, 20, 35, 50, 70, 100 ],
    'n_estimators': [100, 500, 1000, 1250, 1500]
}

gridforest = GridSearchCV(regr, parameters, cv=3, n_jobs=-1)
gridforest.fit(X_train, y_train)

gridforest.best_params_