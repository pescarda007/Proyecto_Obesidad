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

        x['CALC'] = x['CALC'].replace({'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}).astype(int)
        
        x['MTRANS'] = x['MTRANS'].replace({'Walking': 0, 'Bike': 1, 'Motorbike': 2, 'Automobile': 3, 'Public_Transportation':4}).astype(int)
        
        x['NObeyesdad'] = x['NObeyesdad'].replace({'Insufficient_Weight': 0, 'Normal_Weight': 1, 'Overweight_Level_I': 2, 'Overweight_Level_II': 3, 'Obesity_Type_I':4, 'Obesity_Type_II':5, 'Obesity_Type_III':6}).astype(int)
        
        return x