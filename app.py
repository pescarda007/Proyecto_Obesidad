import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.preprocessing import Process as pre
import streamlit as st
from src.preprocessing_utils import BMICalculator, CustomReplacer

ruta = "models/modelo_keras_pipe.pkl"
model=joblib.load(ruta)


def main():
    st.title("Test de obesidad")
    st.text("Este test es interactivo, no es necesario recargar para obtener otros datos")
    # Entrada de datos numéricos
    edad = st.number_input("Edad", min_value=1, max_value=120, step=1)
    altura = st.number_input("Altura (m)", value=0.0, step=0.1, format="%.2f")
    peso = st.number_input("Peso (kg)", value=0.0, step=0.1, format="%.2f")
    
    # Selectbox para opciones específicas
    calc = st.selectbox("Frecuencia consumo de alcohol", ["Nunca", "A veces", "Frecuentemente", "A diario"])
    mtrans = st.selectbox("Medio de transporte que más uses", ["Automóvil", "Bicicleta", "Motocicleta", "Transporte público", "A pie"])
    
    # Otros valores numéricos
    fcvc = st.slider("Consumo de vegetales (1-3)", 1, 3, 2)
    ncp = st.slider("Número de comidas al día", 1, 6, 3)
    faf = st.slider("Actividad física por semana (0-3)", 0, 3, 1)
    
    # Checkbox
    favc = st.checkbox("¿Consumo de comida hipercalórica?")
    smoke = st.checkbox("¿Fumas?")
    historial_familiar = st.checkbox("Historial Familiar de obesidad")
    
    
    # Botón de envío
    if st.button("Enviar"):
        #st.success("Datos enviados correctamente")
        # Cálculo de IMC
        if altura > 0:
            imc = peso / (altura ** 2)
        #Recoger datos
        datos = {
            "Age": edad,
            "Height": round(altura, 2),
            "Weight": round(peso, 2),
            "family_history_with_overweight": historial_familiar,
            "FAVC": favc,
            "FCVC": fcvc,
            "NCP": ncp,
            "SMOKE": smoke,
            "FAF": faf,
            "CALC": calc,
            "MTRANS": mtrans,
            "IMC": round(imc, 2)
            } 
        # Convertir el diccionario a un DataFrame de pandas
        df = pd.DataFrame([datos])
        
        #Preprocesamiento
        df = pre.preprocess_pipe(df)
        data2 = {
            'Age': 0.191489,
            'Height': 0.226415,
            'Weight': 0.298507,
            'family_history_with_overweight': 1.0,
            'FAVC': 1.0,
            'FCVC': 0.5,
            'NCP': 0.0,
            'SMOKE': 0.0,
            'FAF': 0.0,
            'CALC': 0.0,
            'IMC': 0.503764,
            'MTRANS_Bike': 0.0,
            'MTRANS_Motorbike': 0.0,
            'MTRANS_Public_Transportation': 1.0,
            'MTRANS_Walking': 0.0
}

        #df = pd.DataFrame([data2])

        # Llamar al modelo y obtener la respuesta
        predict = model(df)
        resultado = pre.postptocess(predict)
        # Mostrar la predicción en la app
        st.success(resultado)
        

if __name__ == "__main__":
    main()