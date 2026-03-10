EDEMPTION AI: Enterprise ESG & Carbon Prescriptive Analytics Platform
Team Name: REDEPMTION
Domain: Data Science and Analytics
Live URL: https://saroai.site (Live Production Server)

1. Executive Summary
Modern organizations generate vast amounts of carbon emission data, but they struggle to convert this raw data into actionable financial and environmental insights.

To solve this, Team REDEMPTION has built a Full-Stack, AI-powered Enterprise Carbon Platform. Unlike traditional static dashboards, our system utilizes Prescriptive Analytics. It predicts future emissions, calculates financial risks (Carbon Tax penalties), and automatically simulates optimal "What-If" scenarios to tell decision-makers exactly what actions to take to save money and reduce their carbon footprint.

2. Key Platform Features
AI Prescriptive Insight Engine: The platform doesn't just predict pollution; it prescribes solutions. Based on user inputs, the AI recommends specific operational shifts (e.g., "Reduce Diesel by 50% and shift to Renewables") and dynamically displays the Potential Money Saved and Carbon Reduced.
Financial Impact (Carbon Tax Calculator): Translates environmental metrics into business metrics. It calculates the exact financial penalty owed (in USD) if an organization crosses its 12 MT emission threshold.
Enterprise ESG Framework: Categorizes data into industry-standard frameworks: Scope 1 (Direct/Fuel), Scope 2 (Grid Electricity), and Scope 3 (Supply Chain/Travel).
Production-Grade Cloud Deployment: Hosted on a secure, SSL-encrypted cloud server with a modern microservices architecture.
3. Technical Architecture & Tech Stack
We built a production-ready system divided into four main layers:

A. Data Engineering (Synthetic ESG Data)
Since real corporate carbon data is highly confidential, we developed a sophisticated Python script to engineer a 100,000-row enterprise dataset.

Realistic Physics: Electricity consumption dynamically spikes during Summer based on temperature variations (HVAC loads).
Sector Specifics: Modeled varying emission factors for IT (high Grid usage) vs. Manufacturing (high Diesel/Gas usage).
B. The AI Brain (Machine Learning)
Algorithm: Scikit-Learn RandomForestRegressor.
Function: Trained to predict Total_Carbon_Emission_MT with high accuracy based on 9 operational features (Area, Employees, Fuel, Temperature, etc.).
Serialization: Model and encoders saved via joblib.
C. The Backend (REST API)
Framework: FastAPI & Uvicorn (Python).
Logic: Receives JSON payloads from the frontend, processes them through the ML model, calculates Carbon Tax, runs a background optimization simulation, and returns prescriptive insights.
D. The Frontend UI
Tech: HTML5, Tailwind CSS, Vanilla JavaScript.
Design: A sleek, responsive, single-page application (SPA) where users can tweak facility parameters and instantly view their predicted Carbon Emissions without reloading the page.
4. Cloud Infrastructure & DevOps (Deployment)
Moving beyond localhost, the platform is fully deployed on the open web using modern DevOps practices:

Cloud Provider: DigitalOcean Droplet (Ubuntu 24.04 LTS).
Web Server / Reverse Proxy: Nginx. We configured Nginx to serve the static HTML frontend while securely routing API requests (/predict) to the internal FastAPI server running on port 8000.
Process Management: Tmux is used to keep the FastAPI uvicorn worker running persistently in the background.
Security: Secured with an SSL/TLS Certificate generated via Let’s Encrypt (Certbot), ensuring fully encrypted HTTPS communication on the custom domain: saroai.site.
5. Alignment with the Problem Statement
Hackathon Requirement	How REDEMPTION Solved It
"Analyze Emission Sources"	Modeled data using real-world corporate frameworks: Scope 1, 2, and 3 categorization.
"Analyze Trends"	Engineered 100,000 rows of temporal data allowing for deep historical seasonality analysis.
"Reduction Opportunities"	Solved using the AI Prescriptive Simulator. Organizations can test inputs and get automatic AI recommendations on exact ROI and emission reductions.
"Build an Analytics Platform"	Delivered a Full-Stack, cloud-hosted Web Application (saroai.site) with a dedicated AI Backend.
6. Business ROI
By implementing REDEMPTION AI, organizations can:

Transition from reactive reporting to proactive ESG management.
Avoid thousands of dollars in Carbon Tax penalties through intelligent resource allocation.
Achieve corporate Net-Zero targets faster by running cost-free AI simulations before making physical infrastructural changes.
Developed by Team REDEPMTION for the 2026 Data Science & Analytics Hackathon.
