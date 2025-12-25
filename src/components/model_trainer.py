import os
from src.logger import logging
from src.exception import CustomException
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, roc_auc_score, precision_recall_curve
import numpy as np
from dataclasses import dataclass
from src.utils import save_object
import shap 

@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join('artifacts','model.pkl')
    best_params = best_params = {
    'solver': 'saga',
    'l1_ratio': 1.0,          # L1
    'C': 0.09823172642497621,
    'tol': 4.142905343830171e-06,
    'max_iter': 2500,
    'class_weight': 'balanced'
}

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            logging.info('Initializing logistic regression with winning params')
            model = LogisticRegression(**self.model_trainer_config.best_params)

            logging.info('Training Started.')
            model.fit(X_train,y_train)
            logging.info('Training Done.')

            logging.info("Calculating optimal threshold for Recall maximization")
            y_scores = model.predict_proba(X_test)[:, 1]
            _, recalls, thresholds = precision_recall_curve(y_test, y_scores)
            logging.info(f"Using {90}% as target recall score")
            target_recall = 0.90
            valid_indices = np.where(recalls >= target_recall)[0]
            
            if len(valid_indices) > 0:
                best_idx = valid_indices[-1]
                best_idx = min(best_idx, len(thresholds) - 1)
                custom_threshold = thresholds[best_idx]
            else:
                logging.warning("Target recall not reachable. Defaulting to 0.5")
                custom_threshold = 0.5
            
            logging.info(f"Optimal Threshold Found: {custom_threshold:.4f}")

            y_pred_custom = (y_scores >= custom_threshold).astype(int)
            
            final_auc = roc_auc_score(y_test, y_scores)
            final_recall = recall_score(y_test, y_pred_custom)
            
            logging.info(f'Final Model AUC: {final_auc:.4f}, Recall(at threshold {custom_threshold:.2f}): {final_recall:.4f}')

            # print(f"Final Model AUC: {final_auc:.4f}")
            # print(f"Final Model Recall (at threshold {custom_threshold:.2f}): {final_recall:.4f}")
            explainer = shap.LinearExplainer(model,X_train)
            logging.info(f"Saving model and threshold bundle to {self.model_trainer_config.trained_model_path}")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj={
                    "model": model,
                    "threshold": custom_threshold,
                    "explainer":explainer
                }
            )

            return self.model_trainer_config.trained_model_path

        except Exception as e:
            raise CustomException(e)