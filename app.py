import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Enterprise ESG Platform", page_icon="🏢", layout="wide")

# --- LOAD DATA & MODEL ---
@st.cache_data
def load_data():
    return pd.read_csv("Enterprise_ESG_Carbon_Data.csv")

@st.cache_resource
def load_model():
    model = joblib.load('pro_carbon_model.pkl')
    columns = joblib.load('pro_model_columns.pkl')
    return model, columns

df = load_data()
model, model_columns = load_model()

# --- SIDEBAR ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3204/3204094.png", width=80)
st.sidebar.title("🌍 REDEMPTION")
st.sidebar.caption("Enterprise ESG & Carbon AI")
st.sidebar.markdown("---")
menu = st.sidebar.radio("Navigation Menu", ["🏢 Executive Dashboard", "🤖 AI Scenario & ROI Simulator"])

st.sidebar.markdown("---")
st.sidebar.info("💡 **Hackathon Theme:** Sustainability + Environmental Data Science")

# ==========================================
# PAGE 1: EXECUTIVE DASHBOARD
# ==========================================
if menu == "🏢 Executive Dashboard":
    st.title("📊 Global ESG & Carbon Dashboard")
    st.markdown("Track corporate carbon footprint, financial impact (Carbon Tax), and Scope emissions.")

    # Top Level Filters
    company_filter = st.selectbox("Select Company to Analyze", ["All Companies"] + list(df['Company_Name'].unique()))
    
    if company_filter != "All Companies":
        filtered_df = df[df['Company_Name'] == company_filter]
    else:
        filtered_df = df

    # --- KPI METRICS ---
    total_emissions = filtered_df['Total_Carbon_Emission_MT'].sum()
    total_tax = filtered_df['Carbon_Tax_Owed_USD'].sum()
    avg_aqi = filtered_df['Air_Quality_Index_AQI'].mean()
    total_energy_cost = filtered_df['Energy_Cost_USD'].sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Carbon Emissions", f"{total_emissions:,.0f} MT", "Scope 1,2,3")
    col2.metric("⚠️ Total Carbon Tax Owed", f"${total_tax:,.0f}", "-Financial Risk")
    col3.metric("Energy Operational Cost", f"${total_energy_cost:,.0f}")
    col4.metric("Avg Facility AQI", f"{avg_aqi:.0f}", "Environmental Health")

    st.markdown("---")

    # --- CHARTS SECTION ---
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("1. Emissions Breakdown (Scope 1, 2, 3)")
        # Summing the scopes
        s1 = filtered_df['Scope_1_Direct_Emissions'].sum()
        s2 = filtered_df['Scope_2_Indirect_Emissions'].sum()
        s3 = filtered_df['Scope_3_SupplyChain_Emissions'].sum()
        
        scope_data = pd.DataFrame({
            'Scope': ['Scope 1 (Direct - Diesel/Gas)', 'Scope 2 (Indirect - Grid)', 'Scope 3 (Supply Chain/Travel)'],
            'Emissions (kg)': [s1, s2, s3]
        })
        fig_donut = px.pie(scope_data, names='Scope', values='Emissions (kg)', hole=0.5, 
                           color_discrete_sequence=['#ff9999','#66b3ff','#99ff99'])
        st.plotly_chart(fig_donut, use_container_width=True)

    with c2:
        st.subheader("2. Carbon Footprint by City")
        city_df = filtered_df.groupby('Facility_City')['Total_Carbon_Emission_MT'].sum().reset_index()
        fig_bar = px.bar(city_df, x='Facility_City', y='Total_Carbon_Emission_MT', 
                         color='Facility_City', text_auto='.2s')
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("3. Historical Emission Trends & Seasonality")
    filtered_df['Date'] = pd.to_datetime(filtered_df['Date'])
    trend_df = filtered_df.groupby(filtered_df['Date'].dt.to_period('M'))['Total_Carbon_Emission_MT'].sum().reset_index()
    trend_df['Date'] = trend_df['Date'].astype(str)
    fig_line = px.line(trend_df, x='Date', y='Total_Carbon_Emission_MT', markers=True, 
                       line_shape='spline', color_discrete_sequence=['#FF4B4B'])
    st.plotly_chart(fig_line, use_container_width=True)


# ==========================================
# PAGE 2: AI SIMULATOR (The Hackathon Winner)
# ==========================================
elif menu == "🤖 AI Scenario & ROI Simulator":
    st.title("🤖 AI Reduction & ROI Simulator")
    st.markdown("Use our Machine Learning model to simulate operational changes and see the **Financial and Environmental impact** instantly.")

    st.warning("🎯 **Goal:** Reduce Grid Electricity & Diesel to lower Carbon Tax!")

    st.write("### ⚙️ Input Facility Parameters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sector = st.selectbox("Industry Sector", df['Industry_Sector'].unique())
        city = st.selectbox("Facility City", df['Facility_City'].unique())
        employees = st.number_input("Active Employees Count", value=500, step=50)

    with col2:
        area = st.number_input("Facility Area (sqft)", value=50000, step=5000)
        temp = st.slider("Average Temperature (°C)", 10.0, 45.0, 32.0)
        grid_kwh = st.number_input("🔌 Grid Electricity Used (kWh)", value=15000.0, step=1000.0)

    with col3:
        renewables = st.number_input("🌿 Renewable Energy Purchased (kWh)", value=1000.0, step=1000.0)
        diesel = st.number_input("⛽ Diesel Consumed (Liters)", value=500.0, step=50.0)
        gas = st.number_input("🔥 Natural Gas (Therms)", value=50.0, step=10.0)

    st.markdown("---")
    
    if st.button("🔮 Run AI Simulation", type="primary", use_container_width=True):
        
        # Prepare input for ML
        input_data = pd.DataFrame(columns=model_columns)
        input_data.loc[0] = 0 
        
        input_data['Active_Employees_Count'] = employees
        input_data['Total_Facility_Area_sqft'] = area
        input_data['Average_Temperature_C'] = temp
        input_data['Grid_Electricity_kWh'] = grid_kwh
        input_data['Renewable_Energy_Purchased_kWh'] = renewables
        input_data['Diesel_Consumed_Liters'] = diesel
        input_data['Natural_Gas_Therms'] = gas
        
        if f'Industry_Sector_{sector}' in model_columns:
            input_data[f'Industry_Sector_{sector}'] = 1
        if f'Facility_City_{city}' in model_columns:
            input_data[f'Facility_City_{city}'] = 1

        # Predict
        predicted_mt = model.predict(input_data)[0]
        
        # Financial Logic (Assume $40 tax for every MT above 12 MT threshold)
        tax_threshold = 12.0
        tax_owed = max(0, (predicted_mt - tax_threshold) * 40)

        # Display Results
        st.success("✅ AI Prediction Generated Successfully!")
        
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.info("### 🌍 Predicted Carbon Emission")
            st.title(f"{predicted_mt:,.2f} Metric Tons")
            
        with res_col2:
            if tax_owed > 0:
                st.error("### 💸 Estimated Carbon Tax Penalty")
                st.title(f"${tax_owed:,.2f} USD")
            else:
                st.success("### 💸 Estimated Carbon Tax Penalty")
                st.title("$0.00 USD (Tax Free!)")
        
        # Actionable AI Insights
        st.write("### 💡 AI Executive Recommendation:")
        if diesel > 300:
            st.warning(f"**Action Required:** High diesel consumption ({diesel} L) is heavily inflating Scope 1 emissions. **Suggestion:** Shift 50% of logistics/generators to EV/Battery backup to save Carbon Tax.")
        elif grid_kwh > (renewables * 4):
            st.warning(f"**Action Required:** Your facility runs mostly on dirty Grid power. **Suggestion:** Increase 'Renewable Energy Purchased' to at least {grid_kwh * 0.5} kWh in the simulator above and click predict to see your tax drop!")
        else:
            st.success("**Great Job!** Your facility operates efficiently with a good mix of clean energy. Maintain current operations.")