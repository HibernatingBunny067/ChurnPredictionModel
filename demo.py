import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd
import numpy as np

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

with st.sidebar:
    st.header("Project Details")
    st.markdown("""
    **Architecture:**
    - 🧠 Model: Logistic Regression
    - ⚙️ Backend: FastAPI (Code in Repo)
    - 🐳 Container: Docker
    """)
    st.link_button("View Source Code on GitHub", "https://github.com/HibernatingBunny067/ChurnPredictionModel")

st.title("📡 Telco Customer Churn System with Explainable AI")
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
        submitted = st.form_submit_button("🚀 Predict Churn Risk", width='stretch')

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
                shap_vals = result['shap']
                column_names = result['column_names']
                # --- 5. VISUALIZATION ---
                st.divider()
                
                # Create two columns: Metrics on Left, Chart on Right
                vis_col1, vis_col2 = st.columns([0.5,0.5])
                
                with vis_col1:
                    st.subheader("Prediction Details")
                    if pred == 1:
                        st.error(f"🚨 **High Churn Risk Detected**\n\nThe model predicts this customer is likely to leave.")
                    else:
                        st.success(f"✅ **Customer is Safe**\n\nThe model predicts this customer will stay.")
                    
                    # Metrics
                    m1, m2 = st.columns(2)
                    m1.metric("Risk Probability", f"{prob:.3%}", delta_color="inverse")
                    m2.metric("Threshold", f"{thresh*100:.3f}%",help='Dynamically determined from the test test internally')

                with vis_col2:
                    # st.divider()
                    st.subheader("🔍 Why did the system predict this?",help='service provider must aim to work on the RED coloured features')
                    st.write("The chart below shows which factors pushed the customer towards Churn (Red) or Safety (Green).")

                    shap_array = np.array(shap_vals)
                    
                    if shap_array.ndim > 1:
                        shap_array = shap_array[0]
                
                    shap_df = pd.DataFrame({
                        'Feature': column_names,
                        'SHAP Value': shap_array
                    })
                    
                    # Sort by Absolute Impact (Magnitude) to show most important first
                    shap_df['Magnitude'] = shap_df['SHAP Value'].abs()
                    shap_df = shap_df.sort_values('Magnitude', ascending=True).tail(10) # Top 10 factors

                    fig_shap = go.Figure()

                    fig_shap.add_trace(go.Bar(
                        y=shap_df['Feature'],
                        x=shap_df['SHAP Value'],
                        orientation='h',
                        marker=dict(
                            color=shap_df['SHAP Value'].apply(lambda x: '#FF4B4B' if x > 0 else '#2ECC71')
                        ),
                        text=shap_df['SHAP Value'].apply(lambda x: f"{x:+.2f}"), # Show value
                        textposition='outside'
                    ))

                    fig_shap.update_layout(
                        title="<b>Top 10 Factors Driving This Prediction</b>",
                        xaxis_title="Impact on Churn Risk (+ increases risk, - decreases risk)",
                        yaxis_title=None,
                        height=500,
                        showlegend=False,
                        margin=dict(l=0, r=0, t=40, b=0)
                    )

                    st.plotly_chart(fig_shap, width='content')

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

        use_sample_data = st.checkbox('Click here to use sample data',help='Synthetically generated data will be processed.')

        if use_sample_data:
            uploaded_file = 'sample/sample.csv'
        else:
            uploaded_file = st.file_uploader('Upload CSV',type=['csv'])

        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(df)} customers")
            st.dataframe(df.head(5)) # Show preview
            
            if st.button("Process Batch"):
                try:
                    with st.spinner("Processing Batch..."):
                        #request library needs File Like object (byte stream not strings)
                        if use_sample_data:
                            try:
                                file_to_send = open('sample/sample.csv', 'rb')
                            except Exception as e:
                                raise st.error(str(e))
                        else:
                            uploaded_file.seek(0)
                            file_to_send = uploaded_file
                        
                        # Send file to FastAPI
                        files = {"file": ("filename.csv", file_to_send, "text/csv")}
                        with st.spinner('Connecting to the inference API....',):
                            response = requests.post("http://127.0.0.1:8000/predictbatch", files=files)
                            if use_sample_data:
                                file_to_send.close()
                    if response.status_code == 200:
                        results = response.json()
                        print(results)
                        preds = pd.DataFrame(results)
                        print(preds)
                        result_df = pd.concat([df,preds],axis=1)
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

                    st.divider()
                    st.subheader("📊 Batch Prediction Summary")
                    if 'prediction' in result_df.columns:

                        counts = result_df['prediction'].value_counts()
                        labels = [ "Churn Risk" if idx == 1 else "Safe" for idx in counts.index]
                        values = counts.values
                        
                        color_map = {1: '#FF4B4B', 0: '#2ECC71'}
                        colors = [color_map.get(idx, '#888888') for idx in counts.index]

                        # 2. Create Donut Chart
                        fig_batch = go.Figure(data=[go.Pie(
                            labels=labels,
                            values=values,
                            hole=.5, # Makes it a Donut
                            marker=dict(colors=colors),
                            textinfo='label+percent',
                            hoverinfo='label+value'
                        )])

                        fig_batch.update_layout(
                            title="Predicted Class Distribution",
                            height=400,
                            showlegend=True
                        )

                        st.plotly_chart(fig_batch, use_container_width=True)
                        
                        c1, c2 = st.columns(2)
                        c1.metric("Total Customers Processed", len(result_df))
                        churn_count = len(result_df[result_df['prediction'] == 1])
                        c2.metric("Identified At-Risk", churn_count, delta=f"{churn_count/len(result_df):.1%} Rate", delta_color="inverse")
                except Exception as e:
                    st.error(f"Connection Error: {e}")
