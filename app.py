# Streamlit App
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load model and encoders
model = joblib.load("random_forest_model.pkl")
funding_encoder = joblib.load("funding_source_encoder.pkl")
sector_encoder = joblib.load("mtef_sector_encoder.pkl")
agency_encoder = joblib.load("implementing_agency_encoder.pkl")

st.title("Project Success Predictor")

# Input form
st.subheader("Enter Project Details")
cost = st.number_input("Total Project Cost (KES)", min_value=0)
duration = st.number_input("Project Duration (Months)", min_value=0)
funding_source = st.selectbox("Funding Source", funding_encoder.classes_)
mtef_sector = st.selectbox("MTEF Sector", sector_encoder.classes_)
agency = st.selectbox("Implementing Agency", agency_encoder.classes_)

# Helper to encode safely
def encode_label(val, encoder):
    try:
        return encoder.transform([val])[0]
    except:
        return 0

# Encode inputs
encoded_input = pd.DataFrame([{
    'total_project_cost_kes': cost,
    'duration_months': duration,
    'funding_source': encode_label(funding_source, funding_encoder),
    'mtef_sector': encode_label(mtef_sector, sector_encoder),
    'implementing_agency': encode_label(agency, agency_encoder)
}])

# Prediction
if st.button("Predict Success"):
    prediction = model.predict(encoded_input)[0]
    confidence = model.predict_proba(encoded_input)[0][prediction]
    label = "Project is likely to succeed" if prediction == 1 else "Project may fail or stall"
    st.success(f"{label} (Confidence: {confidence:.2%})")
