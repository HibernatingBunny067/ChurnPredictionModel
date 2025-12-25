import streamlit as st
import requests
import plotly.graph_objects as go

# --- 1. CONFIGURATION ---
# Page Config (Browser Title and Icon)
st.set_page_config(page_title="Churn Predictor", page_icon="📡", layout="wide")

# Options (Must match your model training)
GENDER_OPTS = ["Female", "Male"]
YES_NO_OPTS = ["Yes", "No"]
INTERNET_OPTS = ["DSL", "Fiber optic", "No"]
CONTRACT_OPTS = ["Month-to-month", "One year", "Two year"]
PAYMENT_OPTS = ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
SERVICE_OPTS = ["Yes", "No", "No internet service"]
PHONE_OPTS = ["Yes", "No", "No phone service"]

# --- 2. HEADER ---
st.title("📡 Telco Customer Churn System")
st.markdown("""
This dashboard connects to a **FastAPI Microservice** running a Logistic Regression model.
Adjust customer details below to assess the probability of churn.
""")

# --- 3. INPUT FORM (Organized in Columns) ---
with st.form("churn_form"):
    
    # Create 3 columns for better layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("👤 Demographics")
        gender = st.selectbox("Gender", GENDER_OPTS)
        senior = st.radio("Senior Citizen", YES_NO_OPTS, horizontal=True)
        partner = st.radio("Partner", YES_NO_OPTS, horizontal=True)
        dependents = st.radio("Dependents", YES_NO_OPTS, horizontal=True)
        tenure = st.slider("Tenure (Months)", 0, 72, 12)

    with col2:
        st.subheader("🛠️ Services")
        phone = st.radio("Phone Service", YES_NO_OPTS, horizontal=True)
        multiple_lines = st.selectbox("Multiple Lines", PHONE_OPTS)
        internet = st.selectbox("Internet Service", INTERNET_OPTS)
        security = st.selectbox("Online Security", SERVICE_OPTS)
        backup = st.selectbox("Online Backup", SERVICE_OPTS)
        device = st.selectbox("Device Protection", SERVICE_OPTS)
        
    with col3:
        st.subheader("💳 Billing & Others")
        tech = st.selectbox("Tech Support", SERVICE_OPTS)
        tv = st.selectbox("Streaming TV", SERVICE_OPTS)
        movies = st.selectbox("Streaming Movies", SERVICE_OPTS)
        contract = st.selectbox("Contract", CONTRACT_OPTS)
        paperless = st.radio("Paperless Billing", YES_NO_OPTS, horizontal=True)
        payment = st.selectbox("Payment Method", PAYMENT_OPTS)
        monthly = st.number_input("Monthly Charges ($)", value=29.85, step=0.05)
        total = st.number_input("Total Charges ($)", value=29.85, step=1.0)

    # Submit Button
    submitted = st.form_submit_button("🚀 Predict Churn Risk", use_container_width=True)

# --- 4. PREDICTION LOGIC ---
if submitted:
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
        "TotalCharges": str(total)
    }

    try:
        with st.spinner("Connecting to AI Model..."):
            response = requests.post("http://127.0.0.1:8000/predictdata", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            pred = result['prediction']
            prob = result['probability']
            thresh = result['threshold_used']

            # --- 5. VISUALIZATION ---
            st.divider()
            
            # Create two columns: Metrics on Left, Chart on Right
            vis_col1, vis_col2 = st.columns([2, 1])
            
            with vis_col1:
                st.subheader("Prediction Details")
                if pred == 1:
                    st.error(f"🚨 **High Churn Risk Detected**\n\nThe model predicts this customer is likely to leave.")
                else:
                    st.success(f"✅ **Customer is Safe**\n\nThe model predicts this customer will stay.")
                
                # Metrics
                m1, m2 = st.columns(2)
                m1.metric("Risk Probability", f"{prob:.1%}", delta_color="inverse")
                m2.metric("Decision Threshold", f"{thresh:.2f}")

            with vis_col2:
                # --- PIE CHART LOGIC ---
                # We show "Risk" (prob) vs "Safe" (1 - prob)
                labels = ['Churn Risk', 'Safe']
                values = [prob, 1 - prob]
                colors = ['#FF4B4B', '#2ECC71'] # Red for Risk, Green for Safe
                
                fig = go.Figure(data=[go.Pie(
                    labels=labels, 
                    values=values, 
                    hole=.6, # Makes it a Donut Chart
                    marker=dict(colors=colors),
                    textinfo='percent', # Show % on the chart
                    hoverinfo='label+percent'
                )])
                
                # Clean up the layout (remove margins so it fits well)
                fig.update_layout(
                    showlegend=False,
                    margin=dict(t=0, b=0, l=0, r=0),
                    height=200,
                )
                
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.error(f"API Error: {response.status_code} - {response.text}")

    except Exception as e:
        st.error(f"Connection Error. Is the backend running? \n\nDetails: {e}")