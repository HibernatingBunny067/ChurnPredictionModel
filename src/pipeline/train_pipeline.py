from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException

if __name__ == "__main__":
    try:
        print(">>>> STAGE 1: Data Ingestion Started")
        obj = DataIngestion()
        train_data_path, test_data_path = obj.initiate_data_ingestion()
        print(f"Ingestion Completed. Data at: {train_data_path}")

        print(">>>> STAGE 2: Data Transformation Started")
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
        print("Transformation Completed.")

        print(">>>> STAGE 3: Model Training Started")
        model_trainer = ModelTrainer()
        model_trainer.initiate_model_trainer(train_arr, test_arr)
        print(">>>> Model Training Completed & Artifacts Saved!")

    except Exception as e:
        print("Training Failed!")
        raise CustomException(e)