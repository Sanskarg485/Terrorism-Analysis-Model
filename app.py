import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import joblib

# ==========================
# 1. Load Pretrained Objects
# ==========================
scaler = joblib.load("C:/Users/Sanskar Gupta/OneDrive/Desktop/Deep Learning/Terrorist Attack Prediction/scaler.pkl")
le = joblib.load("C:/Users/Sanskar Gupta/OneDrive/Desktop/Deep Learning/Terrorist Attack Prediction/label_encoder.pkl")

# Define same model architecture
class TerrorModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TerrorModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# Model parameters
n_features = 7   # year, month, day, nkill, nwound, latitude, longitude
n_classes = len(le.classes_)

# Load model
model = TerrorModel(n_features, n_classes)
model.load_state_dict(torch.load("C:/Users/Sanskar Gupta/OneDrive/Desktop/Deep Learning/Terrorist Attack Prediction/terrorism_model.pth", map_location=torch.device("cpu")))
model.eval()

# ==========================
# 2. Streamlit UI
# ==========================
st.title("🌍 Global Terrorism Attack Type Prediction")
st.write("Enter event details below to predict the **attack type**.")

# User inputs
year = st.number_input("Year", min_value=1970, max_value=2025, value=2010)
month = st.number_input("Month", min_value=1, max_value=12, value=6)
day = st.number_input("Day", min_value=1, max_value=31, value=15)
nkill = st.number_input("Number of People Killed", min_value=0, value=0)
nwound = st.number_input("Number of People Wounded", min_value=0, value=0)
latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=20.0)
longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=77.0)

# Prepare input
features = np.array([[year, month, day, nkill, nwound, latitude, longitude]])
features_scaled = scaler.transform(features)
features_tensor = torch.tensor(features_scaled, dtype=torch.float32)

# Prediction
if st.button("🔎 Predict Attack Type"):
    with torch.no_grad():
        outputs = model(features_tensor)
        pred_class = torch.argmax(outputs, dim=1).item()
        attack_type = le.inverse_transform([pred_class])[0]
        st.success(f"🎯 Predicted Attack Type: **{attack_type}**")
