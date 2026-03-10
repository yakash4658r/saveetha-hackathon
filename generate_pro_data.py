import numpy as np
import pandas as pd

# =========================
# Synthetic Enterprise ESG Carbon Dataset Generator (India)
# Output: Enterprise_ESG_Carbon_Data.csv
# Rows: 100,000
# =========================

rng = np.random.default_rng(7)

# -------------------------
# 1) Date Range
# -------------------------
dates = pd.date_range("2022-01-01", "2023-12-31", freq="D")
n_days = len(dates)  # 730

# -------------------------
# 2) Companies & Sectors (logical mapping)
# -------------------------
company_to_sector = {
    "TechNova IT Solutions": "IT",
    "DigiSphere Cloud Services": "IT",
    "Apex Heavy Manufacturing": "Manufacturing",
    "EcoBuild Cements": "Manufacturing",
    "Bharat Auto Components": "Manufacturing",
    "Swift Global Logistics": "Logistics",
    "CargoWave Express Logistics": "Logistics",
    "RetailMart India": "Retail",
}

companies = np.array(list(company_to_sector.keys()))
sectors = np.array([company_to_sector[c] for c in companies])

# Company-level ESG maturity (drives renewables + efficiency)
company_esg_maturity = {
    "TechNova IT Solutions": 0.62,
    "DigiSphere Cloud Services": 0.72,
    "Apex Heavy Manufacturing": 0.38,
    "EcoBuild Cements": 0.50,
    "Bharat Auto Components": 0.44,
    "Swift Global Logistics": 0.54,
    "CargoWave Express Logistics": 0.48,
    "RetailMart India": 0.58,
}

# Facility counts (sum = 140) -> 140 * 730 = 102,200 daily records, then sample down to 100,000
facility_counts = {
    "TechNova IT Solutions": 20,
    "DigiSphere Cloud Services": 15,
    "Apex Heavy Manufacturing": 30,
    "EcoBuild Cements": 20,
    "Bharat Auto Components": 15,
    "Swift Global Logistics": 15,
    "CargoWave Express Logistics": 10,
    "RetailMart India": 15,
}

# City list
cities = np.array(["Mumbai", "Chennai", "Delhi", "Bangalore", "Pune"])

# Company-specific city preferences (more realistic)
company_city_probs = {
    "TechNova IT Solutions": np.array([0.12, 0.22, 0.10, 0.40, 0.16]),
    "DigiSphere Cloud Services": np.array([0.10, 0.18, 0.10, 0.46, 0.16]),
    "Apex Heavy Manufacturing": np.array([0.34, 0.12, 0.18, 0.10, 0.26]),
    "EcoBuild Cements": np.array([0.28, 0.10, 0.14, 0.08, 0.40]),
    "Bharat Auto Components": np.array([0.26, 0.14, 0.14, 0.10, 0.36]),
    "Swift Global Logistics": np.array([0.26, 0.18, 0.18, 0.14, 0.24]),
    "CargoWave Express Logistics": np.array([0.22, 0.20, 0.20, 0.14, 0.24]),
    "RetailMart India": np.array([0.24, 0.18, 0.18, 0.16, 0.24]),
}

# Sector -> facility type
sector_to_facility_type = {
    "IT": "Office_Campus",
    "Manufacturing": "Industrial_Plant",
    "Logistics": "Distribution_Warehouse",
    "Retail": "Retail_Store",
}

# -------------------------
# 3) Create Facility Master Data (140 facilities)
# -------------------------
facility_rows = []
facility_id_counter = 1

for company, cnt in facility_counts.items():
    sector = company_to_sector[company]
    esg = company_esg_maturity[company]

    # Sector-specific area distributions (sqft)
    if sector == "IT":
        area_low, area_high = 35_000, 300_000
    elif sector == "Manufacturing":
        area_low, area_high = 180_000, 1_400_000
    elif sector == "Logistics":
        area_low, area_high = 120_000, 1_000_000
    else:  # Retail
        area_low, area_high = 8_000, 130_000

    # Sector-specific operating profile
    if sector == "IT":
        shift_choices, shift_probs = np.array([1, 2, 3]), np.array([0.55, 0.35, 0.10])
        op_hours_mu, op_hours_sigma = 12.0, 1.2
    elif sector == "Manufacturing":
        shift_choices, shift_probs = np.array([2, 3]), np.array([0.35, 0.65])
        op_hours_mu, op_hours_sigma = 22.0, 1.0
    elif sector == "Logistics":
        shift_choices, shift_probs = np.array([2, 3]), np.array([0.45, 0.55])
        op_hours_mu, op_hours_sigma = 20.0, 1.4
    else:  # Retail
        shift_choices, shift_probs = np.array([1, 2]), np.array([0.70, 0.30])
        op_hours_mu, op_hours_sigma = 12.5, 1.0

    for _ in range(cnt):
        city = rng.choice(cities, p=company_city_probs[company])

        # Facility area
        area = rng.uniform(area_low, area_high)

        # Energy efficiency index (higher = more efficient, reduces consumption)
        # Driven by ESG maturity + random, clipped.
        eff_index = np.clip(0.80 + 0.25 * esg + rng.normal(0, 0.06), 0.75, 1.15)

        # Average commuting distance (km) is facility-level (city-dependent)
        commute_base = {
            "Mumbai": 18.0,
            "Delhi": 17.0,
            "Bangalore": 15.0,
            "Chennai": 16.0,
            "Pune": 14.0,
        }[city]
        commute_km = np.clip(commute_base + rng.normal(0, 3.0), 6.0, 35.0)

        # EV fleet % (mainly affects logistics & retail last-mile)
        if sector == "Logistics":
            ev_pct = np.clip(0.10 + 0.55 * esg + rng.normal(0, 0.08), 0.05, 0.85)
        elif sector == "Retail":
            ev_pct = np.clip(0.08 + 0.45 * esg + rng.normal(0, 0.08), 0.02, 0.75)
        else:
            ev_pct = np.clip(0.02 + 0.25 * esg + rng.normal(0, 0.05), 0.00, 0.45)

        # Shifts and operating hours
        shifts = int(rng.choice(shift_choices, p=shift_probs))
        op_hours = float(np.clip(rng.normal(op_hours_mu, op_hours_sigma), 8.0, 24.0))

        facility_rows.append({
            "Facility_ID": f"FAC-{facility_id_counter:04d}",
            "Company_Name": company,
            "Industry_Sector": sector,
            "Facility_City": city,
            "Facility_Type": sector_to_facility_type[sector],
            "Total_Facility_Area_sqft": area,
            "Energy_Efficiency_Index": eff_index,
            "Shift_Count": shifts,
            "Operating_Hours": op_hours,
            "Avg_Commuting_km_per_employee": commute_km,
            "EV_Fleet_Pct": ev_pct,
            "Company_ESG_Maturity": esg,
        })
        facility_id_counter += 1

facilities = pd.DataFrame(facility_rows)

# Employees derived primarily from area + sector density (employees per sqft)
# (Using a noisy density model)
sector_emp_density = {
    "IT": 1 / 115.0,             # 1 employee per 115 sqft
    "Manufacturing": 1 / 290.0,  # 1 per 290 sqft
    "Logistics": 1 / 420.0,      # 1 per 420 sqft
    "Retail": 1 / 190.0,         # 1 per 190 sqft
}
density = facilities["Industry_Sector"].map(sector_emp_density).to_numpy()
base_emp = facilities["Total_Facility_Area_sqft"].to_numpy() * density
emp_noise = rng.lognormal(mean=0.0, sigma=0.18, size=len(facilities))
employees = np.rint(base_emp * emp_noise).astype(int)

# Clamp to realistic sector ranges
def clamp_by_sector(emp_arr, sector_arr):
    emp = emp_arr.astype(float)
    # IT
    m = sector_arr == "IT"
    emp[m] = np.clip(emp[m], 120, 6000)
    # Manufacturing
    m = sector_arr == "Manufacturing"
    emp[m] = np.clip(emp[m], 220, 9000)
    # Logistics
    m = sector_arr == "Logistics"
    emp[m] = np.clip(emp[m], 120, 5000)
    # Retail
    m = sector_arr == "Retail"
    emp[m] = np.clip(emp[m], 40, 1200)
    return emp.astype(int)

facilities["Active_Employees_Base"] = clamp_by_sector(employees, facilities["Industry_Sector"].to_numpy())

# -------------------------
# 4) Expand to Daily Records (Facility x Date)
# -------------------------
idx = pd.MultiIndex.from_product([facilities.index.to_numpy(), dates], names=["Facility_Row", "Date"])
df = idx.to_frame(index=False)
df["Date"] = pd.to_datetime(df["Date"])

# Join facility attributes
df = df.merge(
    facilities.reset_index(drop=True).reset_index().rename(columns={"index": "Facility_Row"}),
    on="Facility_Row",
    how="left"
).drop(columns=["Facility_Row"])

# -------------------------
# 5) Date Features (seasonality, weekends, fiscal)
# -------------------------
dt = df["Date"]
month = dt.dt.month.to_numpy()
doy = dt.dt.dayofyear.to_numpy()
dow = dt.dt.dayofweek.to_numpy()
is_weekend = (dow >= 5).astype(int)

df["DayOfWeek"] = dow
df["Is_Weekend"] = is_weekend
df["Month"] = month
df["Quarter"] = dt.dt.quarter
df["Year"] = dt.dt.year
df["Days_In_Month"] = dt.dt.days_in_month

is_summer = np.isin(month, [4, 5, 6]).astype(int)
is_monsoon = np.isin(month, [7, 8, 9]).astype(int)
is_festival = np.isin(month, [10, 11]).astype(int)  # festival / peak demand season

df["Is_Summer_AprJun"] = is_summer
df["Is_Monsoon_JulSep"] = is_monsoon
df["Is_Festival_Season_OctNov"] = is_festival

# Time trend (renewables adoption, efficiency improvements) from 2022->2023
# scaled 0..1 across entire range
time_trend = ((dt - dates.min()) / (dates.max() - dates.min())).to_numpy().astype(float)
df["Time_Trend_0_1"] = np.clip(time_trend, 0.0, 1.0)

# -------------------------
# 6) City-based Temperature Model (seasonal + city specific)
# -------------------------
city = df["Facility_City"].to_numpy()

city_base = np.select(
    [city == "Mumbai", city == "Chennai", city == "Delhi", city == "Bangalore", city == "Pune"],
    [28.0,            29.0,             26.0,           24.0,               25.0],
    default=26.0
)

city_amp = np.select(
    [city == "Mumbai", city == "Chennai", city == "Delhi", city == "Bangalore", city == "Pune"],
    [4.0,             3.0,              10.0,           3.2,               5.0],
    default=5.0
)

# Phase shift so peak tends toward May (doy ~ 135)
phase = np.select(
    [city == "Mumbai", city == "Chennai", city == "Delhi", city == "Bangalore", city == "Pune"],
    [118.0,           110.0,            125.0,          112.0,              120.0],
    default=118.0
)

seasonal1 = np.sin(2 * np.pi * (doy - phase) / 365.25)
seasonal2 = 0.9 * np.sin(4 * np.pi * (doy - phase) / 365.25)

# Monsoon cooling for coastal cities
monsoon_cool = np.where((city == "Mumbai") | (city == "Chennai"), 1.2, 0.6) * is_monsoon

# Delhi winter extra cold dip (Jan) + mild effect (Dec)
winter_dip = np.where(city == "Delhi", 5.5, 1.8) * np.exp(-((doy - 15) / 28.0) ** 2)

temp_noise = rng.normal(0, 1.4, size=len(df))
temp_c = city_base + city_amp * seasonal1 + seasonal2 - monsoon_cool - winter_dip + temp_noise
temp_c = np.clip(temp_c, 8.0, 46.0)

df["Average_Temperature_C"] = np.round(temp_c, 2)

# -------------------------
# 7) Air Quality Index (AQI) (city baseline + winter spikes + noise)
# -------------------------
aqi_base = np.select(
    [city == "Delhi", city == "Mumbai", city == "Pune", city == "Chennai", city == "Bangalore"],
    [180.0,          145.0,            115.0,          110.0,              95.0],
    default=120.0
)

# Winter pollution tends to spike in North (Delhi) and to a lesser degree Mumbai
winter_months = np.isin(month, [11, 12, 1, 2]).astype(int)
winter_spike = np.where(city == "Delhi", 85.0, np.where(city == "Mumbai", 25.0, 12.0)) * winter_months

# Monsoon cleans air a bit
monsoon_clean = np.where(is_monsoon == 1, 18.0, 0.0)

aqi_noise = rng.normal(0, 22.0, size=len(df))
aqi = aqi_base + winter_spike - monsoon_clean + aqi_noise
aqi = np.clip(aqi, 35.0, 520.0)

df["Air_Quality_Index_AQI"] = np.rint(aqi).astype(int)

# -------------------------
# 8) Operational Metrics (daily employees + production)
# -------------------------
sector = df["Industry_Sector"].to_numpy()
company = df["Company_Name"].to_numpy()

# Employees fluctuate slightly daily; weekends lower for IT/Retail
emp_base = df["Active_Employees_Base"].to_numpy().astype(float)

weekday_attendance = np.where(
    sector == "IT",
    np.where(is_weekend == 1, 0.40, 0.92),
    np.where(sector == "Retail",
             np.where(is_weekend == 1, 0.85, 0.95),
             np.where(sector == "Logistics",
                      np.where(is_weekend == 1, 0.78, 0.96),
                      np.where(is_weekend == 1, 0.82, 0.97)))
)

attendance_noise = rng.normal(0, 0.03, size=len(df))
active_employees = np.rint(emp_base * np.clip(weekday_attendance + attendance_noise, 0.25, 1.05)).astype(int)
active_employees = np.maximum(active_employees, 0)

df["Active_Employees_Count"] = active_employees

# Production volume units (Manufacturing/Logistics only; 0 for IT & Retail)
area = df["Total_Facility_Area_sqft"].to_numpy().astype(float)
op_hours = df["Operating_Hours"].to_numpy().astype(float)
shifts = df["Shift_Count"].to_numpy().astype(float)
eff_idx = df["Energy_Efficiency_Index"].to_numpy().astype(float)

# Base production intensity per sqft per day
mfg_intensity = 0.012  # units per sqft/day baseline
log_intensity = 0.008  # shipments per sqft/day baseline

# Seasonal multipliers:
# - Logistics spikes during festival season (Oct-Nov) and Q4
# - Manufacturing slightly steadier, mild uplift in festival season
log_season_mult = 1.0 + 0.22 * is_festival + 0.08 * np.isin(month, [12, 1]).astype(int)
mfg_season_mult = 1.0 + 0.10 * is_festival - 0.05 * is_monsoon

# Company multipliers (Apex largest/heaviest)
company_prod_mult = np.ones(len(df))
company_prod_mult[company == "Apex Heavy Manufacturing"] *= 1.35
company_prod_mult[company == "EcoBuild Cements"] *= 1.15
company_prod_mult[company == "Bharat Auto Components"] *= 1.05
company_prod_mult[company == "Swift Global Logistics"] *= 1.18
company_prod_mult[company == "CargoWave Express Logistics"] *= 1.05

# Operating intensity
op_intensity = (op_hours / 24.0) * (0.85 + 0.10 * (shifts - 1))

prod_units = np.zeros(len(df), dtype=float)
m_mfg = (sector == "Manufacturing")
m_log = (sector == "Logistics")

prod_noise_mfg = rng.lognormal(mean=0.0, sigma=0.18, size=len(df))
prod_noise_log = rng.lognormal(mean=0.0, sigma=0.22, size=len(df))

prod_units[m_mfg] = (
    area[m_mfg] * mfg_intensity * op_intensity[m_mfg] * mfg_season_mult[m_mfg]
    * company_prod_mult[m_mfg] * prod_noise_mfg[m_mfg]
)

prod_units[m_log] = (
    area[m_log] * log_intensity * op_intensity[m_log] * log_season_mult[m_log]
    * company_prod_mult[m_log] * prod_noise_log[m_log]
)

# Clip and round; IT/Retail remain 0
prod_units = np.where((m_mfg | m_log), np.clip(prod_units, 0, None), 0.0)
df["Production_Volume_Units"] = np.rint(prod_units).astype(int)

# -------------------------
# 9) Grid Outage Hours (drives generator diesel)
# -------------------------
outage_city_mu = np.select(
    [city == "Delhi", city == "Mumbai", city == "Chennai", city == "Bangalore", city == "Pune"],
    [0.85,          0.60,           0.70,             0.50,               0.58],
    default=0.65
)
# More outages during monsoon, slightly during peak summer
outage_mu = outage_city_mu + 0.25 * is_monsoon + 0.10 * is_summer
outage = rng.gamma(shape=1.8, scale=np.clip(outage_mu / 1.8, 0.05, 2.5), size=len(df))
outage = np.clip(outage, 0.0, 8.0)

df["Grid_Outage_Hours"] = np.round(outage, 2)

# -------------------------
# 10) Energy & Fuel Consumption
# -------------------------
# Electricity model: correlates with employees, area, temperature, operating hours, sector baseline
# Summer spike: +35% for IT & Retail during Apr-Jun (MUST)
temp = df["Average_Temperature_C"].to_numpy()

# Cooling degree above 24C
cdd = np.clip(temp - 24.0, 0.0, None)

# Base kWh per employee per day by sector
kwh_per_emp = np.select(
    [sector == "IT", sector == "Manufacturing", sector == "Logistics", sector == "Retail"],
    [9.5,          6.0,                  4.8,               6.5],
    default=6.0
)

# Base kWh per sqft per day by sector (lighting/equipment)
kwh_per_sqft = np.select(
    [sector == "IT", sector == "Manufacturing", sector == "Logistics", sector == "Retail"],
    [0.020,        0.030,               0.014,              0.024],
    default=0.020
)

# Production electricity component
kwh_per_unit_mfg = 0.65
kwh_per_unit_log = 0.18

prod = df["Production_Volume_Units"].to_numpy().astype(float)
prod_kwh = np.zeros(len(df), dtype=float)
prod_kwh[m_mfg] = prod[m_mfg] * kwh_per_unit_mfg
prod_kwh[m_log] = prod[m_log] * kwh_per_unit_log

# Weekend reduction for IT/Offices; retail less reduced
weekend_mult = np.ones(len(df))
weekend_mult[(sector == "IT") & (is_weekend == 1)] = 0.62
weekend_mult[(sector == "Retail") & (is_weekend == 1)] = 0.93
weekend_mult[(sector == "Logistics") & (is_weekend == 1)] = 0.86
weekend_mult[(sector == "Manufacturing") & (is_weekend == 1)] = 0.90

# Temperature HVAC uplift sensitivity
hvac_sens = np.select(
    [sector == "IT", sector == "Retail", sector == "Manufacturing", sector == "Logistics"],
    [0.030,        0.028,              0.015,              0.012],
    default=0.018
)

temp_mult = 1.0 + hvac_sens * cdd
temp_mult = np.clip(temp_mult, 0.90, 1.70)

# Summer spike by rule: +35% for IT & Retail during Apr-Jun
summer_spike_mult = np.ones(len(df))
summer_spike_mult[(is_summer == 1) & np.isin(sector, ["IT", "Retail"])] = 1.35

# Company-specific IT compute intensity (drives higher Scope 2 for TechNova)
company_elec_mult = np.ones(len(df))
company_elec_mult[company == "TechNova IT Solutions"] *= 1.18
company_elec_mult[company == "DigiSphere Cloud Services"] *= 1.08
company_elec_mult[company == "Apex Heavy Manufacturing"] *= 1.12
company_elec_mult[company == "EcoBuild Cements"] *= 1.06

# Efficiency reduces consumption (higher index => lower kWh)
# Convert efficiency index into multiplier where 1.0 is neutral
eff_consumption_mult = np.clip(1.08 - (eff_idx - 0.90) * 0.55, 0.78, 1.18)

elec_base = (
    active_employees * kwh_per_emp +
    area * kwh_per_sqft +
    prod_kwh
)

elec_noise = rng.lognormal(mean=0.0, sigma=0.10, size=len(df))
grid_kwh = elec_base * (op_hours / 24.0) * weekend_mult * temp_mult * summer_spike_mult * company_elec_mult * eff_consumption_mult * elec_noise
grid_kwh = np.clip(grid_kwh, 30.0, None)

df["Grid_Electricity_kWh"] = np.round(grid_kwh, 2)

# Renewable energy purchased: share driven by ESG maturity + time trend + city solar potential
esg = df["Company_ESG_Maturity"].to_numpy()

solar_city_boost = np.select(
    [city == "Chennai", city == "Bangalore", city == "Pune", city == "Mumbai", city == "Delhi"],
    [0.07,            0.06,             0.05,          0.03,           0.02],
    default=0.03
)

# Keep TechNova renewables moderate to preserve "high Scope 2" (but still some renewables)
base_renew_share = 0.08 + 0.55 * esg + 0.08 * time_trend + solar_city_boost + rng.normal(0, 0.05, size=len(df))
# Company override for TechNova (moderate renewables) and Apex (low renewables)
base_renew_share = np.where(company == "TechNova IT Solutions", base_renew_share * 0.65, base_renew_share)
base_renew_share = np.where(company == "Apex Heavy Manufacturing", base_renew_share * 0.70, base_renew_share)

renew_share = np.clip(base_renew_share, 0.00, 0.90)
renew_kwh = grid_kwh * renew_share * np.clip(rng.normal(1.0, 0.08, size=len(df)), 0.65, 1.20)
renew_kwh = np.minimum(renew_kwh, grid_kwh)  # cannot exceed consumption for accounting
renew_kwh = np.clip(renew_kwh, 0.0, None)

df["Renewable_Energy_Share"] = np.round(renew_share, 3)
df["Renewable_Energy_Purchased_kWh"] = np.round(renew_kwh, 2)

# Diesel consumed:
# - Logistics: heavy transport (reduced by EV fleet %)
# - Manufacturing: generators + material handling; Apex is highest
# - IT/Retail: mainly backup generators (low)
ev_pct = df["EV_Fleet_Pct"].to_numpy()

# Logistics diesel per unit (liters/shipment); plus a base for yard vehicles
diesel_per_unit_log = np.select(
    [company == "Swift Global Logistics", company == "CargoWave Express Logistics"],
    [0.22, 0.20],
    default=0.21
)
diesel_log = np.zeros(len(df), dtype=float)
diesel_log[m_log] = (prod[m_log] * diesel_per_unit_log[m_log] + (0.020 * area[m_log] / 1000.0)) * (1.0 - 0.85 * ev_pct[m_log])

# Generator diesel roughly scales with electric load during outage
# Approx liters per outage-hour per (kWh/hour) load:
# Convert daily kWh to average kW over operating hours: kW ~ kWh / op_hours
avg_kw = grid_kwh / np.clip(op_hours, 6.0, 24.0)
gen_lph_per_kw = np.select(
    [sector == "IT", sector == "Retail", sector == "Logistics", sector == "Manufacturing"],
    [0.055,        0.060,           0.070,            0.080],
    default=0.065
)
# Diesel for generator = outage_hours * avg_kw * factor; IT kept low
gen_diesel = outage * avg_kw * gen_lph_per_kw
gen_diesel = np.where(sector == "IT", gen_diesel * 0.35, gen_diesel)

# Manufacturing handling diesel (forklifts etc.)
mfg_handling = np.zeros(len(df), dtype=float)
mfg_handling[m_mfg] = (0.0045 * area[m_mfg] / 1000.0) * np.clip(rng.normal(1.0, 0.12, size=m_mfg.sum()), 0.6, 1.5)

# Apex extra heavy direct fuel usage (ensures highest Scope 1)
apex_mult = np.where(company == "Apex Heavy Manufacturing", 1.45, 1.0)
ecobuild_mult = np.where(company == "EcoBuild Cements", 1.15, 1.0)

diesel_mfg = (gen_diesel + mfg_handling) * apex_mult * ecobuild_mult

# IT/Retail backup diesel (small)
diesel_office_retail = gen_diesel * np.where(np.isin(sector, ["IT", "Retail"]), 1.0, 0.0)

diesel_total = np.zeros(len(df), dtype=float)
diesel_total[m_log] = diesel_log[m_log] + gen_diesel[m_log] * 0.35
diesel_total[m_mfg] = diesel_mfg[m_mfg]
diesel_total[np.isin(sector, ["IT", "Retail"])] = diesel_office_retail[np.isin(sector, ["IT", "Retail"])]

# Force TechNova near-zero Scope 1 by bounding diesel very low for TechNova
diesel_total = np.where(company == "TechNova IT Solutions", np.minimum(diesel_total, 8.0), diesel_total)

diesel_total *= np.clip(rng.normal(1.0, 0.10, size=len(df)), 0.7, 1.35)
diesel_total = np.clip(diesel_total, 0.0, None)
df["Diesel_Consumed_Liters"] = np.round(diesel_total, 2)

# Natural gas therms (manufacturing heating/process). IT/Logistics/Retail ~ 0
natgas = np.zeros(len(df), dtype=float)

# Therms per unit baseline; cement more thermal intense
therm_per_unit = np.where(company == "EcoBuild Cements", 0.115,
                  np.where(company == "Apex Heavy Manufacturing", 0.095,
                  np.where(company == "Bharat Auto Components", 0.080, 0.085)))

# Winter heating uplift (small in India, but visible)
winter_heat_mult = 1.0 + 0.08 * np.isin(month, [12, 1, 2]).astype(int)

natgas[m_mfg] = prod[m_mfg] * therm_per_unit[m_mfg] * winter_heat_mult[m_mfg] * np.clip(rng.normal(1.0, 0.14, size=m_mfg.sum()), 0.6, 1.6)

# Apex must dominate Scope 1 -> add extra process gas for Apex only
natgas += np.where(company == "Apex Heavy Manufacturing", (0.00012 * area) * np.clip(rng.normal(1.0, 0.20, size=len(df)), 0.6, 1.8), 0.0)

# Force TechNova almost zero natural gas
natgas = np.where(company == "TechNova IT Solutions", 0.0, natgas)

natgas = np.clip(natgas, 0.0, None)
df["Natural_Gas_Therms"] = np.round(natgas, 2)

# Water consumption (kL): manufacturing high; cement highest; depends on production + temperature
water = np.zeros(len(df), dtype=float)

water_base = np.select(
    [sector == "IT", sector == "Retail", sector == "Logistics", sector == "Manufacturing"],
    [10.0,         8.0,            18.0,            45.0],
    default=15.0
)
water_area_term = (area / 1000.0) * np.select(
    [sector == "IT", sector == "Retail", sector == "Logistics", sector == "Manufacturing"],
    [0.02,         0.03,           0.05,             0.08],
    default=0.04
)

water_prod_term = np.zeros(len(df), dtype=float)
water_prod_term[m_mfg] = prod[m_mfg] * np.where(company[m_mfg] == "EcoBuild Cements", 0.085, 0.045)
water_prod_term[m_log] = prod[m_log] * 0.008

# Hotter -> more cooling water
water_temp_mult = 1.0 + 0.015 * np.clip(temp - 26.0, 0.0, None)
water_temp_mult = np.clip(water_temp_mult, 0.95, 1.35)

water_noise = rng.lognormal(mean=0.0, sigma=0.16, size=len(df))
water = (water_base + water_area_term + water_prod_term) * water_temp_mult * (op_hours / 24.0) * water_noise
water = np.clip(water, 0.5, None)

df["Water_Consumption_kL"] = np.round(water, 2)

# -------------------------
# 11) Waste, Recycling, Travel (Scope 3 drivers) - extra realism
# -------------------------
waste = np.zeros(len(df), dtype=float)

# Waste from employees + production
waste_emp = active_employees * np.select(
    [sector == "IT", sector == "Retail", sector == "Logistics", sector == "Manufacturing"],
    [0.18,         0.16,           0.14,             0.20],   # kg/employee/day
    default=0.16
)
waste_prod = np.zeros(len(df), dtype=float)
waste_prod[m_mfg] = prod[m_mfg] * np.where(company[m_mfg] == "EcoBuild Cements", 0.55, 0.28)
waste_prod[m_log] = prod[m_log] * 0.06

waste = (waste_emp + waste_prod) * np.clip(rng.normal(1.0, 0.20, size=len(df)), 0.55, 1.8)
waste = np.clip(waste, 0.0, None)
df["Waste_Generated_kg"] = np.round(waste, 2)

# Recycling rate influenced by ESG maturity
recycle = np.clip(0.18 + 0.62 * esg + 0.06 * time_trend + rng.normal(0, 0.08, size=len(df)), 0.05, 0.92)
df["Waste_Recycled_pct"] = np.round(recycle, 3)

# Business travel km/day (IT higher; weekday higher)
travel_base = np.select(
    [sector == "IT", sector == "Retail", sector == "Logistics", sector == "Manufacturing"],
    [0.55,         0.22,           0.28,             0.18],
    default=0.25
)
# km per employee per day (aggregated) -> total travel km
travel_km = active_employees * travel_base * np.where(is_weekend == 1, 0.35, 1.0) * np.clip(rng.lognormal(0.0, 0.35, size=len(df)), 0.3, 5.0)

# Festival season can increase travel for Retail/Logistics
travel_km *= (1.0 + 0.18 * is_festival * np.isin(sector, ["Retail", "Logistics"]).astype(int))

df["Business_Travel_km"] = np.round(travel_km, 2)

# -------------------------
# 12) Scope 1, 2, 3 Emissions (kg CO2e)
# -------------------------
# Emission factors (synthetic but plausible)
EF_DIESEL_KG_PER_L = 2.68
EF_NATGAS_KG_PER_THERM = 5.30
EF_GRID_KG_PER_KWH = 0.72

# Scope 1: Diesel + Natural Gas (as requested)
scope1 = diesel_total * EF_DIESEL_KG_PER_L + natgas * EF_NATGAS_KG_PER_THERM

# Ensure TechNova Scope 1 is almost zero (guardrail)
scope1 = np.where(company == "TechNova IT Solutions", np.minimum(scope1, 25.0), scope1)

df["Scope_1_Direct_Emissions"] = np.round(scope1, 2)

# Scope 2: net grid electricity after renewables (renewables assumed 0 emissions)
net_grid_kwh = np.clip(grid_kwh - renew_kwh, 0.0, None)
scope2 = net_grid_kwh * EF_GRID_KG_PER_KWH
df["Net_Grid_Electricity_kWh"] = np.round(net_grid_kwh, 2)
df["Scope_2_Indirect_Emissions"] = np.round(scope2, 2)

# Scope 3: supply chain / travel / commuting / waste, driven by production & employees
commute_km = df["Avg_Commuting_km_per_employee"].to_numpy()
# Commuting emissions factor (kg CO2e per km per employee-day, blended transport)
EF_COMMUTE_KG_PER_KM = 0.12
# Business travel factor (kg CO2e per km; blended air/rail/road)
EF_TRAVEL_KG_PER_KM = 0.15
# Waste to landfill factor (kg CO2e per kg non-recycled)
EF_WASTE_KG_PER_KG = 0.70

commute_em = active_employees * commute_km * EF_COMMUTE_KG_PER_KM * np.where(is_weekend == 1, 0.35, 1.0)
travel_em = travel_km * EF_TRAVEL_KG_PER_KM
waste_em = waste * (1.0 - recycle) * EF_WASTE_KG_PER_KG

# Procurement/supply chain factor per production unit; IT/Retail per-employee procurement baseline
proc_factor = np.zeros(len(df), dtype=float)
proc_factor[m_mfg] = np.where(company[m_mfg] == "EcoBuild Cements", 12.0,
                      np.where(company[m_mfg] == "Apex Heavy Manufacturing", 10.5,
                      np.where(company[m_mfg] == "Bharat Auto Components", 9.0, 9.5)))
proc_factor[m_log] = np.where(company[m_log] == "Swift Global Logistics", 3.2, 2.9)

proc_em = np.zeros(len(df), dtype=float)
proc_em[m_mfg] = prod[m_mfg] * proc_factor[m_mfg]
proc_em[m_log] = prod[m_log] * proc_factor[m_log]

# IT & Retail supply chain baseline (IT hardware/cloud services, retail procurement)
proc_it_retail = np.zeros(len(df), dtype=float)
m_it = (sector == "IT")
m_ret = (sector == "Retail")
proc_it_retail[m_it] = active_employees[m_it] * np.clip(rng.normal(2.8, 0.6, size=m_it.sum()), 1.2, 5.5)
proc_it_retail[m_ret] = active_employees[m_ret] * np.clip(rng.normal(3.6, 0.8, size=m_ret.sum()), 1.5, 7.0)

scope3 = proc_em + proc_it_retail + commute_em + travel_em + waste_em
# Add mild AQI-related inefficiency (proxy for congestion/inefficiencies) -> increases Scope 3 slightly
scope3 *= (1.0 + 0.0006 * np.clip(aqi - 120.0, 0.0, 250.0))

df["Scope_3_SupplyChain_Emissions"] = np.round(scope3, 2)

# Total emissions in metric tons (MT)
total_kg = scope1 + scope2 + scope3
total_mt = total_kg / 1000.0
df["Total_Carbon_Emission_MT"] = np.round(total_mt, 4)

# Carbon intensity (kg per unit where applicable, else per employee proxy)
denom = np.where(prod > 0, prod, np.maximum(active_employees, 1))
df["Carbon_Intensity_kgCO2e_per_unit"] = np.round(total_kg / denom, 4)

# -------------------------
# 13) Financial Impact
# -------------------------
# Carbon tax: $40 per MT over threshold (threshold varies by sector; synthetic policy)
threshold_mt = np.select(
    [sector == "IT", sector == "Retail", sector == "Logistics", sector == "Manufacturing"],
    [8.0,          10.0,           18.0,              28.0],
    default=12.0
)

carbon_tax = np.clip(total_mt - threshold_mt, 0.0, None) * 40.0
df["Carbon_Tax_Owed_USD"] = np.round(carbon_tax, 2)

# Energy cost: Estimated monthly energy bill (USD) based on daily usage scaled by days in month
# Unit costs (synthetic)
COST_GRID_PER_KWH = 0.115
COST_RENEW_PER_KWH = 0.090
COST_DIESEL_PER_L = 1.12
COST_NATGAS_PER_THERM = 0.82
COST_WATER_PER_KL = 0.42

daily_energy_cost = (
    grid_kwh * COST_GRID_PER_KWH +
    renew_kwh * COST_RENEW_PER_KWH +
    diesel_total * COST_DIESEL_PER_L +
    natgas * COST_NATGAS_PER_THERM +
    water * COST_WATER_PER_KL
)

monthly_cost = daily_energy_cost * df["Days_In_Month"].to_numpy()
df["Energy_Cost_USD"] = np.round(monthly_cost, 2)

# -------------------------
# 14) Enforce critical ML rules / guardrails
# -------------------------
# Rule: Apex must have the highest Scope 1 emissions (ensure by scaling Apex diesel+gas if needed)
apex_mask = df["Company_Name"].to_numpy() == "Apex Heavy Manufacturing"
if apex_mask.any():
    max_scope1_by_company = df.groupby("Company_Name")["Scope_1_Direct_Emissions"].max()
    top_company = max_scope1_by_company.idxmax()
    if top_company != "Apex Heavy Manufacturing":
        # Increase Apex diesel and gas by a factor and recompute Scope 1/Total/Tax/Intensity
        scale = 1.35
        diesel_total = df["Diesel_Consumed_Liters"].to_numpy()
        natgas = df["Natural_Gas_Therms"].to_numpy()

        

        df["Diesel_Consumed_Liters"] = np.round(diesel_total, 2)
        df["Natural_Gas_Therms"] = np.round(natgas, 2)

        scope1 = diesel_total * EF_DIESEL_KG_PER_L + natgas * EF_NATGAS_KG_PER_THERM
        scope1 = np.where(df["Company_Name"].to_numpy() == "TechNova IT Solutions", np.minimum(scope1, 25.0), scope1)
        df["Scope_1_Direct_Emissions"] = np.round(scope1, 2)

        # Recompute totals dependent on Scope 1
        total_kg = scope1 + df["Scope_2_Indirect_Emissions"].to_numpy() + df["Scope_3_SupplyChain_Emissions"].to_numpy()
        total_mt = total_kg / 1000.0
        df["Total_Carbon_Emission_MT"] = np.round(total_mt, 4)

        denom = np.where(df["Production_Volume_Units"].to_numpy() > 0, df["Production_Volume_Units"].to_numpy(), np.maximum(df["Active_Employees_Count"].to_numpy(), 1))
        df["Carbon_Intensity_kgCO2e_per_unit"] = np.round(total_kg / denom, 4)

        carbon_tax = np.clip(total_mt - threshold_mt, 0.0, None) * 40.0
        df["Carbon_Tax_Owed_USD"] = np.round(carbon_tax, 2)

# Rule: TechNova high Scope 2 but almost zero Scope 1 already enforced by diesel/gas bounds.
# Rule: Summer Apr-Jun grid spike +35% for IT/Retail already applied via multiplier 1.35.
# Rule: High renewables => low scope2 ensured by Scope 2 = (Grid - Renew) * EF.

# -------------------------
# 15) Select exactly 100,000 rows, shuffle, final formatting
# -------------------------
# Start from full daily dataset (~102,200 rows) then sample without replacement to exactly 100k.
df = df.sample(n=100_000, random_state=7).reset_index(drop=True)
# Clean-up / column ordering
# Ensure required core columns appear early
core_cols = [
    "Date", "Company_Name", "Industry_Sector", "Facility_City",
    "Facility_ID", "Facility_Type",
    "Active_Employees_Count", "Production_Volume_Units", "Total_Facility_Area_sqft",
    "Average_Temperature_C", "Air_Quality_Index_AQI",
    "Grid_Electricity_kWh", "Renewable_Energy_Purchased_kWh", "Diesel_Consumed_Liters", "Natural_Gas_Therms", "Water_Consumption_kL",
    "Scope_1_Direct_Emissions", "Scope_2_Indirect_Emissions", "Scope_3_SupplyChain_Emissions", "Total_Carbon_Emission_MT",
    "Carbon_Tax_Owed_USD", "Energy_Cost_USD",
]
# Keep all columns; reorder with core first
remaining_cols = [c for c in df.columns if c not in core_cols]
df = df[core_cols + remaining_cols]

# Sort for time-series friendliness
df = df.sort_values(["Date", "Company_Name", "Facility_ID"]).reset_index(drop=True)

# Save
df.to_csv("Enterprise_ESG_Carbon_Data.csv", index=False)