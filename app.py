import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

st.set_page_config(page_title="Donor Project Success Predictor", layout="wide")
st.title("Donor-Funded Project Success Predictor in Kenya")

# File upload
uploaded_file = st.file_uploader("Upload your donor project dataset (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding="ISO-8859-1", skiprows=2)
    df.columns = [col.strip().replace(" ", "_").lower() for col in df.columns]

    # Preview data
    st.subheader("Raw Dataset")
    st.write(df.head())

    # Preprocess
    df['success'] = df['implementation_status'].apply(lambda x: 1 if str(x).strip().lower() == 'complete' else 0)
    df['total_project_cost_(kes)'] = (
        df['total_project_cost_(kes)']
        .str.replace(',', '', regex=False)
        .str.extract(r'(\d+)', expand=False)
        .astype(float)
    )
    df['duration_(months)'] = (
        df['duration_(months)']
        .str.extract(r'(\d+)', expand=False)
        .astype(float)
    )

    features = ['total_project_cost_(kes)', 'funding_type', 'funding_source',
                'duration_(months)', 'mtef_sector', 'implementing_agency']

    data = df[features + ['success']].dropna()
    data = pd.get_dummies(data, columns=['funding_type', 'funding_source', 'mtef_sector', 'implementing_agency'], drop_first=True)

    # Train/Test Split
    X = data.drop('success', axis=1)
    y = data['success']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Selection
    model_option = st.selectbox("Select Model to Train", ("Logistic Regression", "Random Forest", "XGBoost"))

    if model_option == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_option == "Random Forest":
        model = RandomForestClassifier()
    else:
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Model Performance
    st.subheader("Model Performance")
    st.text(classification_report(y_test, preds))

    # Accuracy Comparison Plot
    st.subheader("\U0001F4CA Interactive Accuracy Comparison")
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": RandomForestClassifier(max_depth=5),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    model_scores = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        acc = round(accuracy_score(y_test, preds) * 100, 2)
        model_scores[name] = acc

    score_df = pd.DataFrame(list(model_scores.items()), columns=["Model", "Accuracy (%)"])
    score_df = score_df.sort_values(by="Accuracy (%)", ascending=True)

    fig = px.bar(
        score_df,
        x="Accuracy (%)",
        y="Model",
        orientation='h',
        text=score_df["Accuracy (%)"].apply(lambda x: f"{x:.2f}%"),
        title="Accuracy Comparison of ML Models",
        color="Model",
        color_discrete_sequence=px.colors.sequential.Viridis
    )

    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis_range=[0, 105],
        xaxis_title="Accuracy (%)",
        yaxis_title=None,
        template="plotly_white",
        height=450
    )

    st.plotly_chart(fig, use_container_width=True)

    # Feature Importance
    st.subheader("Feature Importance")
    if model_option == "Logistic Regression":
        coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_[0]})
        st.write(coef_df.sort_values(by='Coefficient', key=abs, ascending=False).head(10))
    else:
        importances = model.feature_importances_
        feat_df = pd.Series(importances, index=X.columns).sort_values(ascending=False).head(10)
        st.bar_chart(feat_df)

    # Exploratory Visualizations
    st.subheader("Exploratory Visualizations")
    col1, col2 = st.columns(2)

    with col1:
        st.write("Project Status Count")
        st.bar_chart(df['implementation_status'].value_counts())

    with col2:
        st.write("Average Project Cost by Sector")
        avg_cost = df.groupby('mtef_sector')['total_project_cost_(kes)'].mean().dropna().sort_values(ascending=False).head(10)
        st.bar_chart(avg_cost)
