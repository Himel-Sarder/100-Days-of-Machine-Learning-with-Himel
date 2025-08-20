import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Polynomial Regression App", layout="wide")
st.title("👟 Shoe Price Prediction (Polynomial Regression)")
st.write("This app predicts **shoe price** based on age, shoe size, and sex using Polynomial Regression.")

# ---------------------------
# File Uploader
# ---------------------------
uploaded_file = st.file_uploader("📂 Upload kaggle.csv", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file, sep=";", encoding="unicode_escape")
    df.rename(columns={'price(£)': 'price'}, inplace=True)

    # Encode categorical
    encoder = LabelEncoder()
    df['sex'] = encoder.fit_transform(df['sex'])

    # Feature engineering
    df['age_to_shoe_size_ratio'] = df['age'] / df['shoe_size']
    df['age_and_price_interaction'] = df['age'] * df['price']

    # ---------------------------
    # Show Dataset
    # ---------------------------
    if st.checkbox("Show Dataset"):
        st.dataframe(df)

    # Pairplot Visualization
    if st.checkbox("Show Pair Plot"):
        st.write("### Pair Plot of Dataset")
        fig = sns.pairplot(df, hue="sex", palette="viridis", height=2)
        st.pyplot(fig)

    # ---------------------------
    # Train Model
    # ---------------------------
    X = df[['age', 'shoe_size', 'sex', 'age_to_shoe_size_ratio', 'age_and_price_interaction']]
    y = df['price']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    degree = st.radio("Select Polynomial Degree:", [2, 3])

    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    y_pred = model.predict(X_test_poly)

    # Model performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader("Model Performance")
    st.write(f"**Degree {degree} Polynomial Regression**")
    st.write(f"- Mean Squared Error: `{mse:.4f}`")
    st.write(f"- R² Score: `{r2:.4f}`")

    # ---------------------------
    # User Prediction
    # ---------------------------
    st.subheader("Try Your Own Prediction")

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", int(df['age'].min()), int(df['age'].max()), 10)
        shoe_size = st.slider("Shoe Size", int(df['shoe_size'].min()), int(df['shoe_size'].max()), 30)
    with col2:
        sex = st.radio("Sex", ["Male", "Female"])

    # Encode input
    sex_encoded = 1 if sex == "Male" else 0
    age_to_shoe_ratio = age / shoe_size
    age_and_price_inter = age * 1  # base assumption for interaction

    # Prepare input
    user_input = pd.DataFrame([[age, shoe_size, sex_encoded, age_to_shoe_ratio, age_and_price_inter]],
                              columns=['age', 'shoe_size', 'sex', 'age_to_shoe_size_ratio', 'age_and_price_interaction'])

    user_input_poly = poly.transform(user_input)
    prediction = model.predict(user_input_poly)[0]

    st.success(f"Predicted Shoe Price: **£{prediction:.2f}**")

else:
    st.warning("Please upload the dataset to continue.")
