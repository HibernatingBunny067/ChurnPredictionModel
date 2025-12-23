import gradio as gr
import requests

# --- 1. CONFIGURATION ---
# These options must match EXACTLY what your model was trained on.
GENDER_OPTS = ["Female", "Male"]
YES_NO_OPTS = ["Yes", "No"]
INTERNET_OPTS = ["DSL", "Fiber optic", "No"]
CONTRACT_OPTS = ["Month-to-month", "One year", "Two year"]
PAYMENT_OPTS = ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
SERVICE_OPTS = ["Yes", "No", "No internet service"] 
PHONE_OPTS = ["Yes", "No", "No phone service"]

def predict_api(gender, senior, partner, dependents, tenure, phone, multiple_lines, internet, 
                security, backup, device, tech, tv, movies, contract, paperless, 
                payment, monthly, total):
    

    payload = {
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0, 
        "Partner": partner,
        "Dependents": dependents,
        "tenure": int(tenure),
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": device,
        "TechSupport": tech,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": float(monthly),
        "TotalCharges": str(total) # Pydantic expects String for TotalCharges
    }

    print("\n--- Sending Payload ---")
    print(payload)

    try:
        response = requests.post("http://127.0.0.1:8000/predictdata", json=payload)
        
        if response.status_code != 200:
            return f"⚠️ API Error {response.status_code}: {response.text}"
            
        result = response.json()

        risk = "🔴 HIGH CHURN RISK" if result['prediction'] == 1 else "🟢 CUSTOMER SAFE"
        prob = f"Probability: {result['probability']:.1%}"
        thresh = f"Threshold Used: {result['threshold_used']:.2f}"
        
        return f"{risk}\n{prob}\n({thresh})"
        
    except Exception as e:
        return f"Connection Error: {str(e)}"


with gr.Blocks(title="Telco Churn Prediction", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📡 Customer Churn Predictor")
    gr.Markdown("Adjust the values below to see how the risk score changes.")
    
    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("### 👤 Demographics")
            # Defaults are important! We set value=... to avoid 'null' errors
            gender = gr.Dropdown(GENDER_OPTS, label="Gender", value="Female")
            senior = gr.Radio(YES_NO_OPTS, label="Senior Citizen", value="No")
            partner = gr.Radio(YES_NO_OPTS, label="Partner", value="No")
            dependents = gr.Radio(YES_NO_OPTS, label="Dependents", value="No")
            tenure = gr.Slider(0, 72, label="Tenure (Months)", value=12)

        with gr.Column(scale=1, min_width=300):
            gr.Markdown("### 🛠️ Services")
            phone = gr.Radio(YES_NO_OPTS, label="Phone Service", value="Yes")
            multiple_lines = gr.Dropdown(PHONE_OPTS, label="Multiple Lines", value="No")
            internet = gr.Dropdown(INTERNET_OPTS, label="Internet Service", value="DSL")
            security = gr.Dropdown(SERVICE_OPTS, label="Online Security", value="No")
            backup = gr.Dropdown(SERVICE_OPTS, label="Online Backup", value="No")
            device = gr.Dropdown(SERVICE_OPTS, label="Device Protection", value="No")
            tech = gr.Dropdown(SERVICE_OPTS, label="Tech Support", value="No")
            tv = gr.Dropdown(SERVICE_OPTS, label="Streaming TV", value="No")
            movies = gr.Dropdown(SERVICE_OPTS, label="Streaming Movies", value="No")

        with gr.Column(scale=1, min_width=300):
            gr.Markdown("### 💳 Billing Info")
            contract = gr.Dropdown(CONTRACT_OPTS, label="Contract", value="Month-to-month")
            paperless = gr.Radio(YES_NO_OPTS, label="Paperless Billing", value="Yes")
            payment = gr.Dropdown(PAYMENT_OPTS, label="Payment Method", value="Electronic check")
            monthly = gr.Number(label="Monthly Charges ($)", value=29.85)
            total = gr.Number(label="Total Charges ($)", value=29.85)
            
            # Predict Button
            btn = gr.Button("🚀 Predict Risk", variant="primary", size="lg")
            output = gr.Textbox(label="Prediction Result", lines=4)

    # Click Event - Pass ALL 19 inputs in correct order
    btn.click(
        fn=predict_api,
        inputs=[
            gender, senior, partner, dependents, tenure, phone, multiple_lines, internet, 
            security, backup, device, tech, tv, movies, contract, paperless, 
            payment, monthly, total
        ],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch(server_port=8080)