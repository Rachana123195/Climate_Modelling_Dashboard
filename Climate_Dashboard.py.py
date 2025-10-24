import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

st.title("🌍 Climate Change Modeling Dashboard")

df = pd.read_csv(r"C:\Users\racha\OneDrive\Documents\Climate_Dashboard.py\climate_nasa (2).csv")
st.subheader("Dataset Preview")
st.dataframe(df.head())

target = st.selectbox("Select Target Variable", df.select_dtypes(include=['number']).columns)
features = st.multiselect("Select Feature Columns", [c for c in df.columns if c != target])

if len(features) > 0:
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_type = st.selectbox("Select Model Type", ["Random Forest", "Linear Regression"])

    if st.button("Train Model"):
        if model_type == "Random Forest":
            model = RandomForestRegressor(random_state=42)
        else:
            model = LinearRegression()

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        st.write("### Model Evaluation Metrics")
        st.write(f"**MAE:** {mae:.4f}")
        st.write(f"**MSE:** {mse:.4f}")
        st.write(f"**R²:** {r2:.4f}")

        result_df = pd.DataFrame({"Actual": y_test, "Predicted": preds})
        st.subheader("Actual vs Predicted")
        st.dataframe(result_df.head())

        joblib.dump(model, "climate_model.pkl")
        st.success("Model trained and saved as climate_model.pkl")

st.subheader("Visualization")
numeric_cols = df.select_dtypes(include=['number']).columns
x_col = st.selectbox("X-axis", numeric_cols)
y_col = st.selectbox("Y-axis", numeric_cols)
st.scatter_chart(df, x=x_col, y=y_col)