import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Load dataset (adjust path if needed)
df = pd.read_csv("C:/Users/Sanskar Gupta/OneDrive/Desktop/Deep Learning/Terrorist Attack Prediction/globalterrorismdb_0718dist.csv", encoding="latin1", low_memory=False)

# Use the same features as model training
features = ['iyear', 'imonth', 'iday', 'nkill', 'nwound', 'latitude', 'longitude']
X = df[features].fillna(0)

# Target column
y = df['attacktype1_txt'].dropna()

# Fit scaler and label encoder
scaler = StandardScaler().fit(X)
le = LabelEncoder().fit(y)

# Save them
joblib.dump(scaler, "C:/Users/Sanskar Gupta/OneDrive/Desktop/Deep Learning/Terrorist Attack Prediction/scaler.pkl")
joblib.dump(le, "C:/Users/Sanskar Gupta/OneDrive/Desktop/Deep Learning/Terrorist Attack Prediction/label_encoder.pkl")

print("✅ Scaler and LabelEncoder saved successfully!")
