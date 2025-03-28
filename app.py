import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from preprocessing import Process as pre
import streamlit as st

ruta = "modelo_keras.pkl"
model=joblib.load(ruta)


def main():
    st.title("Formulario de Datos Personales")
    
    # Entrada de datos numéricos
    edad = st.number_input("Edad", min_value=0, max_value=120, step=1)
    altura = st.number_input("Altura (cm)", min_value=50, max_value=250, step=1)
    peso = st.number_input("Peso (kg)", min_value=10, max_value=300, step=1)
    
    # Checkbox para historial familiar
    historial_familiar = st.checkbox("Historial Familiar de Enfermedades")
    
    # Selectbox para opciones específicas
    smoke = st.selectbox("¿Fumas?", ["No", "Ocasionalmente", "Frecuentemente", "Sí"])
    calc = st.selectbox("Consumo de alcohol", ["Nunca", "A veces", "Frecuentemente", "Diario"])
    mtrans = st.selectbox("Medio de transporte", ["Automóvil", "Bicicleta", "Motocicleta", "Transporte público", "A pie"])
    
    # Otros valores numéricos
    favc = st.slider("Consumo de comida hipercalórica (1-5)", 1, 5, 3)
    fcvc = st.slider("Consumo de vegetales (1-5)", 1, 5, 3)
    ncp = st.slider("Número de comidas al día", 1, 10, 3)
    faf = st.slider("Actividad física por semana (horas)", 0, 20, 3)
    
    # Cálculo de IMC
    if altura > 0:
        imc = peso / ((altura / 100) ** 2)
    
    print(f"Edad {edad}")
    print(f"Altura {altura}")
    print(f"Peso  {peso}")
    print(f"Antecedente familiar  {historial_familiar}")
    print(f"Fuma {smoke}")
    print(f"Alcohol {calc}")
    print(f"Transporte {mtrans}")
    print(f"Consumo comida hipercalorica {favc}")
    print(f"Consumo vegetales {fcvc}")
    print(f"Comidas del día {ncp}")
    print(f"Actividad física {faf}")
    
    # Botón de envío
    if st.button("Enviar"):
        st.success("Datos enviados correctamente")
        datos = st.json({
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
        })
        
        df = pd.DataFrame([datos])
        
        #Preprocesamiento
        df = pre.preprocess()
         # Llamar al modelo y obtener la respuesta
        resultado = model(df)
        
        # Mostrar la predicción en la app
        st.success(resultado)

if __name__ == "__main__":
    main()