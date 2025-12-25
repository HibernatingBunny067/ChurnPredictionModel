import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd

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
st.logo(image='images/img.jpeg',size='large')
st.title("📡 Telco Customer Churn System")
st.markdown("""
This dashboard connects to a **FastAPI Microservice** running a Logistic Regression model.
Adjust customer details or upload csv below to assess the probability of churn.
""")

tab1,tab2 = st.tabs(['Single Prediction','Batched Prediction'])
with tab1:
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
            with st.spinner('Connecting to the inference API....'):
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
                    m2.metric("Decision Threshold (identified in experimentation for maximum recall)", f"{thresh:.2f}")

                with vis_col2:
                    labels = ['Churn Risk', 'Safe']
                    values = [prob, 1 - prob]
                    colors = ['#FF4B4B', '#2ECC71'] 
                    
                    fig = go.Figure(data=[go.Pie(
                        labels=labels, 
                        values=values, 
                        hole=.6, 
                        marker=dict(colors=colors),
                        textinfo='percent', 
                        hoverinfo='label+percent'
                    )])
                    
                    fig.update_layout(
                        showlegend=False,
                        margin=dict(t=0, b=0, l=0, r=0),
                        height=200,
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)


                if pred == 1:
                    with st.expander("Recommended Retention Strategy", expanded=True):
                        st.write("To prevent this customer from leaving, consider these actions:")
                        st.markdown("""
                        - 🎁 **Offer a 1-Year Contract discount** (Moves risk from High to Low)
                        - 📞 **Schedule a Tech Support check-up** (If TechSupport was 'No')
                        - 📉 **Review Monthly Charges** (If > $80, offer a downgrade to DSL)
                        """)

            else:
                st.error(f"API Error: {response.status_code} - {response.text}")

        except Exception as e:
            st.error(f"Connection Error. Is the backend running? \n\nDetails: {e}")

    with tab2:
        st.header("Batched Predictions")
        uploaded_file = st.file_uploader('Upload CSV',type=['csv'])

        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(df)} customers")
            st.dataframe(df.head(5)) # Show preview
            
            if st.button("Process Batch"):
                try:
                    with st.spinner("Processing Batch..."):
                        # RESET file pointer to beginning before sending
                        uploaded_file.seek(0)
                        
                        # Send file to FastAPI
                        files = {"file": ("filename.csv", uploaded_file, "text/csv")}
                        with st.spinner('Connecting to the inference API....',):
                            response = requests.post("http://127.0.0.1:8000/predictbatch", files=files)
                    
                    if response.status_code == 200:
                        results = response.json()
                        result_df = pd.DataFrame(results).drop(columns=['row_index'])
                        
                        # Display Results
                        st.success("Batch Processing Complete!")
                        st.dataframe(result_df)
                        
                        # Download Button
                        csv = result_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "Download Results CSV",
                            csv,
                            "churn_predictions.csv",
                            "text/csv",
                            key='download-csv'
                        )
                    else:
                        st.error(f"Error {response.status_code}: {response.text}")
                    
                except Exception as e:
                    st.error(f"Connection Error: {e}")