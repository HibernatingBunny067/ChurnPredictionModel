import numpy as np
import pandas as pd
import random 

def generate_synthetic_data(num_samples=600):
    # 1. Defining options exactly as the model expects them
    genders = ["Female", "Male"]
    yes_no = ["Yes", "No"]
    internet_types = ["DSL", "Fiber optic", "No"]
    contracts = ["Month-to-month", "One year", "Two year"]
    payment_methods = ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    
    # Service options differ based on the Internet service, but for simplicity, 
    # we use the standard set. The preprocessor handles 'No internet service' mapping usually.
    service_opts = ["Yes", "No", "No internet service"]
    phone_opts = ["Yes", "No", "No phone service"]

    data = []

    for _ in range(num_samples):
        # 2. Randomly generate basic details
        tenure = np.random.randint(0, 73) # 0 to 72 months
        monthly_charges = round(np.random.uniform(18.25, 118.75), 2)
        
        # Calculate TotalCharges (approximate relation + some noise)
        if tenure == 0:
            total_charges = " " # Blank string for new customers 
        else:
            # Tenure * Monthly + random noise
            charge = (tenure * monthly_charges) + np.random.uniform(-10, 10)
            total_charges = str(round(max(charge, 0), 2))

        row = {
            "gender": random.choice(genders),
            "SeniorCitizen": random.choice([0, 1]),
            "Partner": random.choice(yes_no),
            "Dependents": random.choice(yes_no),
            "tenure": tenure,
            "PhoneService": random.choice(yes_no),
            "MultipleLines": random.choice(phone_opts),
            "InternetService": random.choice(internet_types),
            "OnlineSecurity": random.choice(service_opts),
            "OnlineBackup": random.choice(service_opts),
            "DeviceProtection": random.choice(service_opts),
            "TechSupport": random.choice(service_opts),
            "StreamingTV": random.choice(service_opts),
            "StreamingMovies": random.choice(service_opts),
            "Contract": random.choice(contracts),
            "PaperlessBilling": random.choice(yes_no),
            "PaymentMethod": random.choice(payment_methods),
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }
        data.append(row)

    # 3. Create DataFrame and Save
    df = pd.DataFrame(data)
    output_filename = "sample/sample.csv"
    df.to_csv(output_filename, index=False)
    print(f"Successfully created {output_filename} with {num_samples} rows!")
    return df

if __name__ == "__main__":
    generate_synthetic_data(600)