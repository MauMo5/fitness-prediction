import numpy as np
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier


st.write(''' # Predicción Fitness ''')
st.image("fitnessimage.png", caption="La predicción es en base a ciertos dates de su día a día")

st.header('Datos de evaluación')

def user_input_features():
  # Entrada
  Edad = st.number_input('Edad:', min_value=1, max_value=100, value = 1, step = 1)
  Estatura = st.number_input('Estatura en cm:', min_value=0, max_value=300, value = 0, step = 1)
  Peso= st.number_input('Peso en kg:', min_value=0, max_value=500, value = 0, step = 1)
  Frecuenciacar = st.number_input('Frecuencia cardíaca:',min_value=0, max_value=200, value = 0, step = 1)
  Presionart = st.number_input('Presión arterial:', min_value=0, max_value=200, value = 0, step = 1)
  Sueño = st.number_input('Horas de sueño:', min_value=0, max_value=24, value = 0, step = 1)
  Calidadnut= st.number_input('Calidad nutritiva: ', min_value=0, max_value=10, value=0, step=1)
  Indiceact= st.number_input('Índice de actividad: ', min_value=0, max_value=5, value=0, step=1)
  Fumar= st.number_input('Fuma 0=no 1=sí: ', min_value=0, max_value=1, value=0, step=1)
  Género= st.number_input('Género 0=mujer 1=hombre: ', min_value=0, max_value=1, value=0, step=1)

  user_input_data = {'age': Edad,
                     'height_cm': Estatura,
                     'weight_kg': Peso,
                     'heart_rate': Frecuenciacar,
                     'blood_pressure': Presionart,
                     'sleep_hours': Sueño,
                     'nutrition_quality': Calidadnut,
                     'activity_index': Indiceact,
                     'smokes': Fumar,
                     'gender': Género
                     }

  features = pd.DataFrame(user_input_data, index=[0])

  return features

df = user_input_features()

fitnessdf =  pd.read_csv('fitnesspred.csv', encoding='latin-1')
X = fitnessdf.drop(columns='is_fit')
Y = fitnessdf['is_fit']

classifier = DecisionTreeClassifier(max_depth=4, criterion='gini', min_samples_leaf=25, max_features=5, random_state=1613797)
classifier.fit(X, Y)

prediction = classifier.predict(df)

st.subheader('Predicción')
if prediction == 0:
  st.write('No es fitness')
elif prediction == 1:
  st.write('Si es fitness')
else:
  st.write('Sin predicción')
