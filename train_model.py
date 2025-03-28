import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from tensorflow.keras import layers

#Cargar archivo
file_path = "F:\Proyecto_Obesidad\ObesityDataSet_raw_and_data_sinthetic.csv"
df = pd.read_csv(file_path)


#procesamiento
df.drop_duplicates()
df["IMC"] = df['IMC'] = df['Weight'] / df['Height']**2

#Scaler
scaler = MinMaxScaler()

df[["Height","Weight","IMC"]] = scaler.fit_transform(df[['Height', 'Weight',"IMC"]])
df[["Age"]] = scaler.fit_transform(df[["Age"]])

df[['family_history_with_overweight', 'FAVC','SMOKE', 'SCC']] = df[['family_history_with_overweight', 'FAVC','SMOKE', 'SCC']].replace({'yes': True, 'no': False})
df[['family_history_with_overweight','FAVC',"SMOKE","SCC"]] = df[['family_history_with_overweight','FAVC',"SMOKE","SCC"]].astype(bool)
df[['family_history_with_overweight','FAVC',"SMOKE","SCC"]]  = df[['family_history_with_overweight','FAVC',"SMOKE","SCC"]].astype(int)

df['Gender'] = df['Gender'].replace({'Male': 0, 'Female': 1})
df['Gender'] = df['Gender'].astype(int)

df[["FCVC","NCP","CH2O","FAF","TUE"]]= df[["FCVC","NCP","CH2O","FAF","TUE"]].astype(int)

df['CAEC'] = df['CAEC'].replace({'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}).astype(int)
df['CALC'] = df['CALC'].replace({'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}).astype(int)

df['MTRANS'] = df['MTRANS'].replace({'Walking': 0, 'Bike': 1, 'Motorbike': 2, 'Automobile': 3, 'Public_Transportation':4}).astype(int)

df['NObeyesdad'] = df['NObeyesdad'].replace({'Insufficient_Weight': 0, 'Normal_Weight': 1, 'Overweight_Level_I': 2, 'Overweight_Level_II': 3, 'Obesity_Type_I':4, 'Obesity_Type_II':5, 'Obesity_Type_III':6}).astype(int)

x = df[["Age","Height","Weight","family_history_with_overweight","FAVC","FCVC","NCP","SMOKE","FAF","CALC","MTRANS","IMC"]]
y = df["NObeyesdad"]

#Datos
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

y_train = to_categorical(y_train,num_classes=7)
y_test = to_categorical(y_test,num_classes=7)


#modelo
num_classes = 7
input_size = x.shape[1]

model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(input_size,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])
#Compilar
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=128, epochs=100,
          verbose=1)