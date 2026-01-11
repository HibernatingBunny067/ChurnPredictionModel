import os
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin,BaseEstimator
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataTransformationConfig():
    preprocessor_obj_file_path:str = os.path.join('artifacts','preprocessor.pkl')

##custom feature engineering class
class FeatureEngineer(BaseEstimator,TransformerMixin):
    def __init__(self):
        self.fill_value = 0
    def fit(self,X,y=None):
        temp_charges = pd.to_numeric(X['TotalCharges'],errors='coerce')
        self.fill_value = temp_charges.mean()
        return self
    def transform(self,X):
        try:
            X_copy = X.copy()
            X_copy['TotalCharges'] = pd.to_numeric(X_copy['TotalCharges'], errors='coerce')
            X_copy['TotalCharges'] = X_copy['TotalCharges'].fillna(self.fill_value)
            X_copy['TotalCharges'] = np.log1p(X_copy['TotalCharges'])
            # --- Feature Engineering Logic from the Notebook ---
            
            # 1. Contract + Payment Interaction
            X_copy['contract_payment'] = X_copy['Contract'] + '_' + X_copy['PaymentMethod']
            
            # 2. Internet + TechSupport Interaction
            X_copy['internet_support'] = X_copy['InternetService'] + '_' + X_copy['TechSupport']
            
            # 3. Family Status
            X_copy['family_status'] = X_copy['Partner'] + '_' + X_copy['Dependents']
            
            # 4. Auto Payment Flag
            X_copy['auto_payment'] = X_copy['PaymentMethod'].apply(lambda x: 1 if 'automatic' in x else 0)
            
            # 5. Tenure Cohorts
            X_copy['tenure_group'] = pd.cut(X_copy['tenure'], bins=[-1, 12, 48, 100], labels=['New', 'Established', 'Loyal'])
            X_copy['tenure_group'] = X_copy['tenure_group'].astype(str)

            return X_copy
            
        except Exception as e:
            raise CustomException(e)


class DataTransformation():
    def __init__(self):
        self.transformation_config = DataTransformationConfig()
    
    def get_data_transformer_obj(self):
        try:
            logging.info('Defining Columns from the dataset')

            numerical_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
        
            categorical_columns = [
                'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 
                'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
                'Contract', 'PaperlessBilling', 'PaymentMethod', 
                # PLUS the new columns created by FeatureEngineer
                'tenure_group', 'contract_payment', 'internet_support', 'family_status'
            ]

            logging.info('Buidling Numerical Pipeline')
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            logging.info("Building Categorical Pipeline")
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ],
                remainder='drop' # Droping columns like customerID
            )
            
            full_pipeline = Pipeline(
                steps=[
                    ("feature_engineer", FeatureEngineer()),
                    ("preprocessor", preprocessor)
                ]
            )

            return full_pipeline
        
        except Exception as e:
            raise CustomException(e)
    
    def initiate_data_transformation(self,train_path,test_path):
        try:

            logging.info("Read train and test data started")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_obj()

            target_column_name = "Churn"
            drop_columns = [target_column_name, "customerID"]

            logging.info("Separating Input Features and Target Features")
            
            # X (Features)
            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)

            # y (targets)
            target_feature_train_df = train_df[target_column_name].map({'Yes': 1, 'No': 0})
            target_feature_test_df = test_df[target_column_name].map({'Yes': 1, 'No': 0})

            logging.info(f"Applying preprocessing object on training and testing dataframes.")
            
            # Fit & Transform Train, Transform Test
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Concatenate X and y (to pass to Model Trainer)
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info(f"Saved preprocessing object, and returned the pickle object file path")

            save_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            return (
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e)
