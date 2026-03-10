import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

def train_enterprise_model():
    print("⏳ Loading Enterprise ESG Dataset (100,000 rows)...")
    file_name = "Enterprise_ESG_Carbon_Data.csv"
    
    if not os.path.exists(file_name):
        print(f"❌ Error: {file_name} not found! Run the data generator first.")
        return

    df = pd.read_csv(file_name)
    
    print("⚙️ Selecting features for the AI Brain...")
    # These are the inputs the AI will look at (Features)
    features = [
        'Industry_Sector', 'Facility_City', 'Active_Employees_Count', 
        'Total_Facility_Area_sqft', 'Average_Temperature_C', 
        'Grid_Electricity_kWh', 'Renewable_Energy_Purchased_kWh', 
        'Diesel_Consumed_Liters', 'Natural_Gas_Therms'
    ]
    
    # This is what the AI needs to predict (Target)
    target = 'Total_Carbon_Emission_MT' # Metric Tons
    
    X = df[features]
    y = df[target]
    
    print("🔠 Converting Text data to Numbers (One-Hot Encoding)...")
    categorical_cols = ['Industry_Sector', 'Facility_City']
    X_encoded = pd.get_dummies(X, columns=categorical_cols)
    
    # Save column names for the Streamlit app later
    joblib.dump(list(X_encoded.columns), 'pro_model_columns.pkl')
    
    print("🔀 Splitting data into Train and Test...")
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    
    print("🧠 Training the Enterprise Random Forest Model... (Wait 10-30 seconds)")
    model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    print("🧪 Testing Accuracy...")
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    
    print("\n✅ --- MODEL RESULTS ---")
    print(f"🎯 Accuracy (R2 Score): {r2 * 100:.2f}%")
    print(f"📉 Average Error: Off by only {mae:.2f} Metric Tons")
    
    # Save the powerful new model
    joblib.dump(model, 'pro_carbon_model.pkl')
    print("\n💾 Model saved as 'pro_carbon_model.pkl'")
    print("🚀 DONE! We are ready to build the Ultimate Streamlit Dashboard.")

if __name__ == "__main__":
    train_enterprise_model()