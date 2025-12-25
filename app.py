from fastapi import FastAPI,HTTPException, UploadFile,File
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
from sklearn.preprocessing import StandardScaler
import uvicorn
from pydantic import BaseModel
import io


class CustomerInput(BaseModel): ##class to take input
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: str

app = FastAPI(title='Churn Prediction App')

@app.get('/')
def index():
    return {'message':'home'}

@app.post('/predictdata')
def predict_datapoint(data:CustomerInput):
    try:
        data_dict = data.model_dump()
        custom_data = CustomData(**data_dict)
        pred_df = custom_data.get_data_as_data_frame()
        print("Data Frame Before Prediction:\n", pred_df) # For debugging

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))

@app.post('/predictbatch')
async def predict_batch(file:UploadFile = File()):
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        predict_pipeline = PredictPipeline()
        
        results = []
    
        for index, row in df.iterrows():
            single_row_df = row.to_frame().T
            
            try:
                prediction = predict_pipeline.predict(single_row_df)
        
                results.append({
                    "row_index": index,
                    "prediction": prediction['prediction'],
                    "probability": prediction['probability'],
                    "risk_level": "High" if prediction['prediction'] == 1 else "Low"
                })
            except Exception as e:
                results.append({"row_index": index, "error": str(e)})

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch Processing Failed: {str(e)}")



if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)