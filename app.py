import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import joblib

# 1. Re-define the Model Architecture
class ActuarialPredictor(nn.Module):
    def __init__(self, input_dim):
        super(ActuarialPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x): return self.network(x)

# 2. Load the trained artifacts
@st.cache_resource
def load_assets():
    model = ActuarialPredictor(input_dim=8)
    model.load_state_dict(torch.load('model.pth', weights_only=True))
    model.eval()
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()

# 3. Build the UI
st.title("🏥 Actuarial Health Insurance Engine")
st.markdown("This machine learning tool prices individual health insurance premiums based on statistical medical risk.")

st.sidebar.header("Patient Profile")
age = st.sidebar.slider("Age", 18, 65, 30)
bmi = st.sidebar.slider("BMI", 15.0, 50.0, 25.0)
children = st.sidebar.slider("Dependents", 0, 5, 0)
sex = st.sidebar.selectbox("Biological Sex", ["Male", "Female"])
smoker = st.sidebar.selectbox("Smoker Status", ["No", "Yes"])
region = st.sidebar.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])

# 4. Map Inputs to Model Format
sex_male = 1 if sex == "Male" else 0
smoker_yes = 1 if smoker == "Yes" else 0
region_northwest = 1 if region == "Northwest" else 0
region_southeast = 1 if region == "Southeast" else 0
region_southwest = 1 if region == "Southwest" else 0

# Feature order must match training exactly
input_data = np.array([[age, bmi, children, sex_male, smoker_yes, region_northwest, region_southeast, region_southwest]])
scaled_input = scaler.transform(input_data)
tensor_input = torch.tensor(scaled_input, dtype=torch.float32)

# 5. Run Inference & Actuarial Math
with torch.no_grad():
    expected_claims = model(tensor_input).item()

loading_factor = 0.20 # 20% margin for admin and risk
loaded_annual_cost = expected_claims * (1 + loading_factor)
monthly_premium = loaded_annual_cost / 12

# 6. Display Results
st.subheader("Actuarial Valuation")
col1, col2, col3 = st.columns(3)

col1.metric("Predicted Annual Claims", f"${expected_claims:,.2f}")
col2.metric("Safety Margin (20%)", f"${(expected_claims * loading_factor):,.2f}")
col3.metric("Final Monthly Premium", f"${monthly_premium:,.2f}", delta="Payable Monthly", delta_color="off")

st.divider()
st.markdown("**Note for Recruiters:** The underlying PyTorch regression model was trained on the Medical Cost Personal Datasets. It isolates compound non-linear risks (e.g., BMI interacting with tobacco usage) to output a pure expected claim cost, to which standard actuarial premium principles are applied.")