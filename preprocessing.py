import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class Process:
    def preprocess(x):
        scaler = MinMaxScaler()

        x[["Height","Weight","IMC"]] = scaler.fit_transform(x[['Height', 'Weight',"IMC"]])
        x[["Age"]] = scaler.fit_transform(x[["Age"]])
        
        x[['family_history_with_overweight', 'FAVC','SMOKE']] = x[['family_history_with_overweight', 'FAVC','SMOKE']].replace({'yes': True, 'no': False})
        x[['family_history_with_overweight','FAVC',"SMOKE"]] = x[['family_history_with_overweight','FAVC',"SMOKE"]].astype(bool)
        x[['family_history_with_overweight','FAVC',"SMOKE"]]  = x[['family_history_with_overweight','FAVC',"SMOKE"]].astype(int)
            
        x[["FCVC","NCP","FAF"]]= x[["FCVC","NCP","FAF"]].astype(int)

        x['CALC'] = x['CALC'].replace({'Nunca': 0, 'A veces': 1, 'Frecuentemente': 2, 'A diario': 3}).astype(int)
        
        x['MTRANS'] = x['MTRANS'].replace({'A pie': 0, 'Bicicleta': 1, 'Motocicleta': 2, 'Automóvil': 3, 'Transporte público':4}).astype(int)
        
        #x['NObeyesdad'] = x['NObeyesdad'].replace({'Insufficient_Weight': 0, 'Normal_Weight': 1, 'Overweight_Level_I': 2, 'Overweight_Level_II': 3, 'Obesity_Type_I':4, 'Obesity_Type_II':5, 'Obesity_Type_III':6}).astype(int)
        
        return x
    
    
    def postptocess(prediccion):
        clases = {
            0 :'Peso insuficiente, estás por debajo de la media',
            1 : 'Peso normal sigue así',
            2 : 'Estás un poco por encima del peso, pero es saludable',
            3 : 'Estás por encima del peso y entrando en una zona peligrosa para la salud',
            4 : 'Obesidad de tipo 1 a darle zapatilla, hay que bajar de peso',
            5 : 'Obesodad de tipo 2',
            6 : 'Obesidad de tipo 3'
        }
        
        valores = prediccion.numpy()
        clase_idx = valores.argmax()
        
        return clases.get(clase_idx)