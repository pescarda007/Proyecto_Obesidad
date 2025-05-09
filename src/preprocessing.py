import pandas as pd
import numpy as np
import joblib
from src.preprocessing_utils import BMICalculator, CustomReplacer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer


class Process:
    def preprocess(x):
        ohe=joblib.load("src/one-hot_encoder.pkl")
        scaler = joblib.load("src/minmaxscaler.pkl")

        x[["Height","Weight","IMC"]] = scaler.fit_transform(x[['Height', 'Weight',"IMC"]])
        x[["Age"]] = scaler.fit_transform(x[["Age"]])
        
        x[['family_history_with_overweight', 'FAVC','SMOKE']] = x[['family_history_with_overweight', 'FAVC','SMOKE']].replace({'yes': True, 'no': False})
        x[['family_history_with_overweight','FAVC',"SMOKE"]] = x[['family_history_with_overweight','FAVC',"SMOKE"]].astype(bool)
        x[['family_history_with_overweight','FAVC',"SMOKE"]]  = x[['family_history_with_overweight','FAVC',"SMOKE"]].astype(int)
            
        x[["FCVC","NCP","FAF"]] = x[["FCVC","NCP","FAF"]].astype(int)

        x['CALC'] = x['CALC'].replace({'Nunca': 0, 'A veces': 1, 'Frecuentemente': 2, 'A diario': 3}).astype(int)
        
        x[['FCVC','NCP','FAF']] = scaler.fit_transform(x[['FCVC','NCP','FAF']])
        
        x['MTRANS'] = x['MTRANS'].replace({'Automóvil': 'Automobile', 'Bicicleta': 'Bike', 'Motocicleta': 'Motobike', 'Transporte público': 'Public_Transportation', 'A pie':'Walking'})
        
        valor_transformado = ohe.transform([x["MTRANS"]])
        x = x.drop('MTRANS', axis=1)
        df = np.concatenate([x, valor_transformado], axis=1)
        return df
    
    
    def postptocess(prediccion):
        clases = {
            0 :'Peso insuficiente, estás por debajo de la media',
            1 :'Peso normal sigue así',
            2 :'Estás un poco por encima del peso, pero es saludable',
            3 :'Estás por encima del peso y entrando en una zona peligrosa para la salud',
            4 :'Obesidad de tipo 1 a darle zapatilla, hay que bajar de peso',
            5 :'Obesodad de tipo 2 zona amarilla hay que ponerse seri@ a bajar ese peso',
            6 :'Obesidad de tipo 3 ZONA ROJA PELIGRO, hipertensión arterial, diabetes mellitus, cardiopatía coronaria, insuficiencia respiratoria y dislipidemia'
        }
        
        valores = prediccion.numpy()
        clase_idx = valores.argmax(axis=1)[0]
        
        return clases.get(clase_idx)
    
    

    def preprocess_pipe(x):
        pipeline = joblib.load("src/pipeline_completo.pkl")
        x[['family_history_with_overweight', 'FAVC','SMOKE']] = x[['family_history_with_overweight', 'FAVC','SMOKE']].replace({True: 'yes', False:'no' })
        x['CALC'] = x['CALC'].replace({'Nunca': 'no', 'A veces': 'Sometimes', 'Frecuentemente': 'Frequently', 'A diario': 'Always'})
        x['MTRANS'] = x['MTRANS'].replace({'Automóvil': 'Automobile', 'Bicicleta': 'Bike', 'Motocicleta': 'Motobike', 'Transporte público': 'Public_Transportation', 'A pie':'Walking'})
        x_pipe = pipeline.transform(x)
        return x_pipe