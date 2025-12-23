import numpy as np
import pandas as pd
import os
import json
from fastapi import FastAPI
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from dataclasses import dataclass

@dataclass
class artifacts_config:
    model_path = 'artifacts/model.pkl'
    preprocessor_path = 'artifacts/preprocessor.pkl'

class PredictPipeline:
    def __init__(self):
        self.config = artifacts_config
        self.model_bundle = load_object(file_path=self.config.model_path)
        self.preprocessor = load_object(file_path=self.config.preprocessor_path)

    def predict(self,features):
        try:
            data_scaled = self.preprocessor.transform(features)
            model = self.model_bundle['model']
            thresh = self.model_bundle['threshold']
            preds = model.predict_proba(data_scaled)[:,1]
            prediction = (preds > thresh)

            results = {
                "prediction": float(prediction),
                "probability": float(preds[0]),
                "threshold_used": float(thresh)
            }
            
            return results
        except Exception as e:
            raise CustomException(e)

class CustomData:
    def __init__(self,
        gender: str,
        SeniorCitizen: int,
        Partner: str,
        Dependents: str,
        tenure: int,
        PhoneService: str,
        MultipleLines: str,
        InternetService: str,
        OnlineSecurity: str,
        OnlineBackup: str,
        DeviceProtection: str,
        TechSupport: str,
        StreamingTV: str,
        StreamingMovies: str,
        Contract: str,
        PaperlessBilling: str,
        PaymentMethod: str,
        MonthlyCharges: float,
        TotalCharges: str  # Note: Can be str if your pipeline handles conversion
    ):
        self.gender = gender
        self.SeniorCitizen = SeniorCitizen
        self.Partner = Partner
        self.Dependents = Dependents
        self.tenure = tenure
        self.PhoneService = PhoneService
        self.MultipleLines = MultipleLines
        self.InternetService = InternetService
        self.OnlineSecurity = OnlineSecurity
        self.OnlineBackup = OnlineBackup
        self.DeviceProtection = DeviceProtection
        self.TechSupport = TechSupport
        self.StreamingTV = StreamingTV
        self.StreamingMovies = StreamingMovies
        self.Contract = Contract
        self.PaperlessBilling = PaperlessBilling
        self.PaymentMethod = PaymentMethod
        self.MonthlyCharges = MonthlyCharges
        self.TotalCharges = TotalCharges

    def get_data_as_data_frame(self):
        """
        Converts the class attributes into a DataFrame.
        This is critical because Scikit-Learn pipelines expect 
        DataFrames with specific column names, not raw values.
        """
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "SeniorCitizen": [self.SeniorCitizen],
                "Partner": [self.Partner],
                "Dependents": [self.Dependents],
                "tenure": [self.tenure],
                "PhoneService": [self.PhoneService],
                "MultipleLines": [self.MultipleLines],
                "InternetService": [self.InternetService],
                "OnlineSecurity": [self.OnlineSecurity],
                "OnlineBackup": [self.OnlineBackup],
                "DeviceProtection": [self.DeviceProtection],
                "TechSupport": [self.TechSupport],
                "StreamingTV": [self.StreamingTV],
                "StreamingMovies": [self.StreamingMovies],
                "Contract": [self.Contract],
                "PaperlessBilling": [self.PaperlessBilling],
                "PaymentMethod": [self.PaymentMethod],
                "MonthlyCharges": [self.MonthlyCharges],
                "TotalCharges": [self.TotalCharges],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e)
        
