# Streamlit App
import streamlit as st
import pandas as pd
import joblib

# Load the saved Random Forest model
model = joblib.load("random_forest_model.pkl")

st.title("Project Success Predictor")

# User inputs
st.subheader("Enter Project Details:")
cost = st.number_input("Total Project Cost (KES)", min_value=0)
duration = st.number_input("Project Duration (Months)", min_value=0)
funding_source = st.selectbox("Funding Source", ["Government of Kenya", "World Bank", "UNICEF", "USAID", "Unknown"])
mtef_sector = st.selectbox("MTEF Sector", ["Education", "Health", "Infrastructure", "Governance", "Unknown"])
agency = st.selectbox("Implementing Agency", ["Ministry of Education", "Ministry of Health", "NGO", "County Government", "Unknown"])

# Encoding (replicate label encoders from training)
def encode_label(val, encoder):
    try:
        return encoder.transform([val])[0]
    except:
        return 0  # fallback to 'Unknown' or index 0
# Encode categorical features
label_encoders = {}
for col in ['funding_source', 'mtef_sector', 'implementing_agency']:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    label_encoders[col] = le
    # Save each encoder
    joblib.dump(le, f"{col}_encoder.pkl")

# Reload encoders if you saved them
funding_encoder = joblib.load("funding_source_encoder.pkl")
sector_encoder = joblib.load("mtef_sector_encoder.pkl")
agency_encoder = joblib.load("implementing_agency_encoder.pkl")

# Encode inputs
encoded_input = pd.DataFrame([{
    'total_project_cost_kes': cost,
    'duration_months': duration,
    'funding_source': encode_label(funding_source, funding_encoder),
    'mtef_sector': encode_label(mtef_sector, sector_encoder),
    'implementing_agency': encode_label(agency, agency_encoder)
}])

# Predict button
if st.button("Predict Success"):
    prediction = model.predict(encoded_input)[0]
    label = "Successful" if prediction == 1 else "Not Successful"
    st.success(f"Prediction: {label}")
