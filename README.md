# MarthaAI_EngineeringCaseStudy - **Ad Performance Analysis** 
**Setup Instructions:**
Open your Terminal then follow these steps:
# 1. Install libraries:
pip install -r requirements.txt

# 2. Run Dashboard
cd frontend <br>
python dashboard.py



# 3. Architecture Overview:
The app is designed with backend folder where the data is generated and the ML model is built.
frontend where the customer can see the dashboard to interact and see the insights.

# 4. Tech Stack Choices:
- The app is built using Python as it has a lot of packages to manipulate the data (ex: numpy and pandas).
On top of that it has a library called Dash plotly to build interactive dashboard with a little bit of configuration. 
- In terms of the model, RandomForestRegressor is used because of High Performance, Handles Nonlinear Relationships and No Need for Feature Scaling. However, I wouldn't use it in case of explainability (it is less interpretable) or Real-time predictions.

# 5. Demo

