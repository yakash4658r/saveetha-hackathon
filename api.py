from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from fastapi.middleware.cors import CORSMiddleware

# 1. Initialize FastAPI App
app = FastAPI(title="Enterprise Carbon AI API", description="API for predicting Carbon Tax & Emissions")

# Enable CORS (Idhu irundha thaan React/HTML la irundhu API call panna mudiyum)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all frontend websites to connect
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Load the AI Model & Columns
print("⏳ Loading AI Model...")
model = joblib.load('pro_carbon_model.pkl')
model_columns = joblib.load('pro_model_columns.pkl')
print("✅ Model Loaded Successfully!")

# 3. Define Input Data Format (What frontend will send)
class FacilityData(BaseModel):
    Industry_Sector: str
    Facility_City: str
    Active_Employees_Count: int
    Total_Facility_Area_sqft: float
    Average_Temperature_C: float
    Grid_Electricity_kWh: float
    Renewable_Energy_Purchased_kWh: float
    Diesel_Consumed_Liters: float
    Natural_Gas_Therms: float

# 4. Create the API Endpoint
@app.post("/predict")
def predict_carbon_emission(data: FacilityData):
    
    # --- ORIGINAL PREDICTION ---
    input_df = pd.DataFrame(columns=model_columns)
    input_df.loc[0] = 0 
    
    input_df['Active_Employees_Count'] = data.Active_Employees_Count
    input_df['Total_Facility_Area_sqft'] = data.Total_Facility_Area_sqft
    input_df['Average_Temperature_C'] = data.Average_Temperature_C
    input_df['Grid_Electricity_kWh'] = data.Grid_Electricity_kWh
    input_df['Renewable_Energy_Purchased_kWh'] = data.Renewable_Energy_Purchased_kWh
    input_df['Diesel_Consumed_Liters'] = data.Diesel_Consumed_Liters
    input_df['Natural_Gas_Therms'] = data.Natural_Gas_Therms

    if f'Industry_Sector_{data.Industry_Sector}' in model_columns:
        input_df[f'Industry_Sector_{data.Industry_Sector}'] = 1
    if f'Facility_City_{data.Facility_City}' in model_columns:
        input_df[f'Facility_City_{data.Facility_City}'] = 1

    predicted_mt = model.predict(input_df)[0]
    tax_threshold = 12.0
    tax_owed = max(0, (predicted_mt - tax_threshold) * 40)
    
    # --- ADVANCED AI REDUCTION OPTIMIZER (The Magic) ---
    # Create a copy of user's data to simulate a better scenario
    optimized_df = input_df.copy()
    
    # AI Logic: Reduce Diesel by 50% and shift 30% of Grid to Solar
    optimized_df['Diesel_Consumed_Liters'] = data.Diesel_Consumed_Liters * 0.5
    optimized_df['Grid_Electricity_kWh'] = data.Grid_Electricity_kWh * 0.7
    optimized_df['Renewable_Energy_Purchased_kWh'] = data.Renewable_Energy_Purchased_kWh + (data.Grid_Electricity_kWh * 0.3)
    
    # Predict for the optimized scenario
    opt_predicted_mt = model.predict(optimized_df)[0]
    opt_tax_owed = max(0, (opt_predicted_mt - tax_threshold) * 40)
    
    # Calculate Savings
    carbon_saved = predicted_mt - opt_predicted_mt
    money_saved = tax_owed - opt_tax_owed

    # Generate Dynamic Actionable Insight Text
    insight_action = f"Switch 50% of your logistics/generators ({int(data.Diesel_Consumed_Liters * 0.5)} Liters) to EV, and convert 30% of grid power to Solar."

    return {
        "status": "success",
        "predictions": {
            "Total_Carbon_Emission_MT": round(predicted_mt, 2),
            "Estimated_Carbon_Tax_USD": round(tax_owed, 2)
        },
        "optimization": {
            "action_required": insight_action,
            "potential_carbon_reduction_MT": round(carbon_saved, 2),
            "potential_money_saved_USD": round(money_saved, 2)
        }
    }

# Health check endpoint
@app.get("/")
def home():
    return {"message": "Carbon AI API is Running Live!"}