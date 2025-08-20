import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Shoe Price Predictor", page_icon="👟", layout="wide")

st.title("Shoe Price Prediction App")
st.markdown("""
This app uses **Polynomial Regression (Degree 2)** to predict shoe prices based on age, shoe size, and gender.  
The model is trained on a small built-in dataset with feature engineering.
""")

# ---------------------------
# Sidebar Inputs
# ---------------------------
st.sidebar.header("🔧 Input Parameters")

def user_input_features():
    age = st.sidebar.slider("Age", 3, 20, 10)
    shoe_size = st.sidebar.slider("Shoe Size", 25, 40, 30)
    sex = st.sidebar.selectbox("Sex", ("Male", "Female"))
    sex_encoded = 1 if sex == "Male" else 0
    return age, shoe_size, sex_encoded, sex

# ---------------------------
# Load Built-in Data
# ---------------------------
@st.cache_data
def load_data():
    data = {
        'age': [3, 4, 5, 6, 7, 8, 9, 9, 10, 11, 11, 12, 10, 11, 12],
        'shoe_size': [27, 27, 28, 29, 29, 31, 30, 30, 29, 32, 31, 34, 32, 31, 31],
        'price': [4, 4, 5, 5, 6, 7, 6, 7, 8, 5, 3, 6, 7, 9, 11],
        'sex': ['m', 'm', 'm', 'f', 'f', 'm', 'f', 'm', 'f', 'm', 'f', 'm', 'm', 'm', 'f']
    }
    df = pd.DataFrame(data)

    # Encode categorical
    encoder = LabelEncoder()
    df['sex'] = encoder.fit_transform(df['sex'])

    # Feature engineering
    df['age_to_shoe_size_ratio'] = df['age'] / df['shoe_size']
    df['age_and_price_interaction'] = df['age'] * df['price']

    return df

df = load_data()

# ---------------------------
# Train Model
# ---------------------------
@st.cache_resource
def train_model(df):
    X = df[['age', 'shoe_size', 'sex', 'age_to_shoe_size_ratio', 'age_and_price_interaction']]
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    return model, poly, X_test_poly, y_test

model, poly_transformer, X_test, y_test = train_model(df)

# ---------------------------
# User Input & Prediction
# ---------------------------
age, shoe_size, sex_encoded, sex_display = user_input_features()

age_to_shoe_size_ratio = age / shoe_size
# start with placeholder interaction (will recalc after prediction)
input_data = np.array([[age, shoe_size, sex_encoded, age_to_shoe_size_ratio, 0]])

# 1st pass prediction
input_poly = poly_transformer.transform(input_data)
predicted_price = model.predict(input_poly)[0]

# update interaction with predicted price
input_data[0, 4] = age * predicted_price
input_poly = poly_transformer.transform(input_data)
final_prediction = model.predict(input_poly)[0]

# ---------------------------
# Display Results
# ---------------------------
st.subheader("Prediction Results")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Age", age)
col2.metric("Shoe Size", shoe_size)
col3.metric("Sex", sex_display)
col4.metric("Predicted Price (£)", f"£{final_prediction:.2f}")

# Model performance
st.subheader("Model Performance")
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
col1, col2 = st.columns(2)
col1.metric("Mean Squared Error", f"{mse:.4f}")
col2.metric("R² Score", f"{r2:.4f}")

# ---------------------------
# Visualizations
# ---------------------------
st.subheader("Data Visualization")
tab1, tab2, tab3 = st.tabs(["Scatter Plots", "Distribution", "Correlation Matrix"])

with tab1:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.scatterplot(data=df, x='age', y='price', hue='sex', ax=axes[0])
    axes[0].set_title('Age vs Price')
    axes[0].scatter(age, final_prediction, color='red', s=100, label='Your Prediction')
    axes[0].legend()

    sns.scatterplot(data=df, x='shoe_size', y='price', hue='sex', ax=axes[1])
    axes[1].set_title('Shoe Size vs Price')
    axes[1].scatter(shoe_size, final_prediction, color='red', s=100, label='Your Prediction')
    axes[1].legend()
    st.pyplot(fig)

with tab2:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.histplot(df['age'], kde=True, ax=axes[0])
    axes[0].axvline(age, color='red', linestyle='--')
    axes[0].set_title('Age Distribution')

    sns.histplot(df['shoe_size'], kde=True, ax=axes[1])
    axes[1].axvline(shoe_size, color='red', linestyle='--')
    axes[1].set_title('Shoe Size Distribution')

    sns.histplot(df['price'], kde=True, ax=axes[2])
    axes[2].axvline(final_prediction, color='red', linestyle='--')
    axes[2].set_title('Price Distribution')
    st.pyplot(fig)

with tab3:
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0, ax=ax)
    st.pyplot(fig)

# ---------------------------
# Raw Data
# ---------------------------
if st.checkbox("Show Raw Data"):
    st.dataframe(df)

# Footer
st.markdown("---")
st.caption("Built with Streamlit | Polynomial Regression (Degree 2) | Feature Engineering")
