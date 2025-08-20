import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Shoe Price Predictor",
    page_icon="👟",
    layout="wide"
)

# Title and description
st.title("Shoe Price Prediction App")
st.markdown("""
This app predicts shoe prices based on age, shoe size, and sex using a polynomial regression model.
Adjust the input parameters using the sliders and select box below.
""")

# Load and prepare the data
@st.cache_data
def load_data():
    # Load the dataset from the provided path
    df = pd.read_csv('kaggle.csv', sep=";", encoding='unicode_escape')
    df.rename(columns={'price(£)': 'price'}, inplace=True)
    
    # Convert categorical 'sex' into numerical values
    encoder = LabelEncoder()
    df['sex'] = encoder.fit_transform(df['sex'])
    
    # Feature engineering
    df['age_to_shoe_size_ratio'] = df['age'] / df['shoe_size']
    df['age_and_price_interaction'] = df['age'] * df['price']
    
    return df, encoder

try:
    df, encoder = load_data()
    
    # Display dataset info
    with st.expander("Show Dataset Info"):
        st.subheader("Dataset Overview")
        st.write(f"Dataset shape: {df.shape}")
        st.dataframe(df)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("Summary Statistics")
            st.write(df.describe())
        with col2:
            st.write("Data Types")
            st.write(df.dtypes.astype(str))
    
    # Train the model
    @st.cache_resource
    def train_model():
        # Define features and target
        X = df[['age', 'shoe_size', 'sex', 'age_to_shoe_size_ratio', 'age_and_price_interaction']]
        y = df['price']
        
        # Create polynomial features (degree 2)
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        
        # Train the model
        model = LinearRegression()
        model.fit(X_poly, y)
        
        return model, poly

    model, poly = train_model()

    # Sidebar for user input
    st.sidebar.header("Input Parameters")

    def user_input_features():
        age = st.sidebar.slider('Age', int(df['age'].min()), int(df['age'].max()) + 5, int(df['age'].mean()))
        shoe_size = st.sidebar.slider('Shoe Size', int(df['shoe_size'].min()), int(df['shoe_size'].max()) + 5, int(df['shoe_size'].mean()))
        sex = st.sidebar.selectbox('Sex', ('Male', 'Female'))
        return age, shoe_size, sex

    age, shoe_size, sex = user_input_features()

    # Convert sex to numerical value
    sex_encoded = 1 if sex == 'Male' else 0

    # Calculate derived features
    age_to_shoe_size_ratio = age / shoe_size
    age_and_price_interaction = age  # This will be updated after prediction

    # Create input array for prediction
    input_data = np.array([[age, shoe_size, sex_encoded, age_to_shoe_size_ratio, 0]])  # placeholder for interaction

    # Transform input data
    input_poly = poly.transform(input_data)

    # Make initial prediction
    prediction = model.predict(input_poly)[0]

    # Update the interaction feature with the predicted price
    age_and_price_interaction = age * prediction
    input_data[0, 4] = age_and_price_interaction

    # Transform again with updated interaction feature
    input_poly = poly.transform(input_data)
    prediction = model.predict(input_poly)[0]

    # Display prediction
    st.subheader("Prediction")
    st.metric("Predicted Shoe Price", f"£{prediction:.2f}")

    # Visualization
    st.subheader("Data Visualization")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Price Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df['price'], kde=True, ax=ax)
        ax.axvline(prediction, color='red', linestyle='--', label=f'Predicted Price: £{prediction:.2f}')
        ax.legend()
        st.pyplot(fig)

    with col2:
        st.write("Age vs Price")
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x='age', y='price', hue='sex', ax=ax)
        ax.scatter(age, prediction, color='red', s=100, label='Prediction')
        ax.legend()
        st.pyplot(fig)
        
    # Additional visualizations
    col3, col4 = st.columns(2)
    
    with col3:
        st.write("Shoe Size vs Price")
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x='shoe_size', y='price', hue='sex', ax=ax)
        ax.scatter(shoe_size, prediction, color='red', s=100, label='Prediction')
        ax.legend()
        st.pyplot(fig)
        
    with col4:
        st.write("Correlation Heatmap")
        fig, ax = plt.subplots()
        numeric_df = df.select_dtypes(include=[np.number])
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    # Model performance evaluation
    st.subheader("Model Performance")
    
    # Split data for evaluation
    X = df[['age', 'shoe_size', 'sex', 'age_to_shoe_size_ratio', 'age_and_price_interaction']]
    y = df['price']
    X_poly = poly.transform(X)
    
    # Calculate metrics
    y_pred = model.predict(X_poly)
    r2 = model.score(X_poly, y)
    mse = np.mean((y_pred - y) ** 2)
    
    col5, col6 = st.columns(2)
    with col5:
        st.metric("R² Score", f"{r2:.4f}")
    with col6:
        st.metric("Mean Squared Error", f"{mse:.4f}")
        
    # Actual vs Predicted plot
    st.write("Actual vs Predicted Prices")
    fig, ax = plt.subplots()
    ax.scatter(y, y_pred, alpha=0.7)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    ax.set_xlabel('Actual Price')
    ax.set_ylabel('Predicted Price')
    ax.set_title('Actual vs Predicted Prices')
    st.pyplot(fig)

    # Model information
    st.subheader("Model Information")
    st.write("This app uses a Polynomial Regression model (Degree 2) with the following features:")
    st.markdown("""
    - Age
    - Shoe Size
    - Sex (encoded as 0 for Female, 1 for Male)
    - Age to Shoe Size Ratio
    - Age and Price Interaction
    """)

except FileNotFoundError:
    st.error("""
    **File Not Found Error**: The dataset file was not found at the specified path.
    
    Please make sure the file path is correct or upload your dataset using the file uploader below.
    """)
    
    # File uploader as fallback
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, sep=";", encoding='unicode_escape')
            st.write("Dataset uploaded successfully!")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error reading the file: {e}")
    else:
        st.info("Please upload a CSV file to continue.")

# Footer
st.markdown("---")
st.markdown("*Note: This is a demonstration app. Predictions are based on a limited dataset and may not be accurate for real-world applications.*")
