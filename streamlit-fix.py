import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained models
@st.cache_resource
def load_models():
    with open('models/linear_regression.pkl', 'rb') as f:
        lr_model = pickle.load(f)
    with open('models/decision_tree.pkl', 'rb') as f:
        dt_model = pickle.load(f)
    with open('models/random_forest.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return lr_model, dt_model, rf_model, scaler

def preprocess_input(input_dict):
    # Create DataFrame with a single row
    df = pd.DataFrame([input_dict])
    
    # Encode categorical variables
    categorical_features = ['company', 'fueltype', 'aspiration', 'doornumber', 'carbody',
                          'drivewheel', 'enginelocation', 'enginetype',
                          'cylindernumber', 'fuelsystem']
    
    label_encoder = LabelEncoder()
    for feature in categorical_features:
        df[feature] = label_encoder.fit_transform(df[feature])
    
    # Ensure columns are in the correct order
    expected_columns = ['wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight',
                       'enginesize', 'boreratio', 'stroke', 'compressionratio', 'horsepower',
                       'peakrpm', 'citympg', 'highwaympg', 'company', 'fueltype', 
                       'aspiration', 'doornumber', 'carbody', 'drivewheel', 'enginelocation',
                       'enginetype', 'cylindernumber', 'fuelsystem']
    
    df = df[expected_columns]
    return df

def main():
    st.title('Car Price Prediction System')
    st.write("""
    ### Project Description
    This application predicts car prices based on various features using different machine learning models.
    """)
    
    # Load models and scaler
    lr_model, dt_model, rf_model, scaler = load_models()
    
    # Sidebar inputs
    st.sidebar.header('User Input Features')
    
    # Collect all inputs in a dictionary
    input_data = {
        'company': st.sidebar.selectbox('Company', 
            ['toyota', 'honda', 'bmw', 'mercedes', 'audi', 'volkswagen']),
        'fueltype': st.sidebar.selectbox('Fuel Type', 
            ['gas', 'diesel']),
        'aspiration': st.sidebar.selectbox('Aspiration', 
            ['std', 'turbo']),
        'doornumber': st.sidebar.selectbox('Number of Doors', 
            ['two', 'four']),
        'carbody': st.sidebar.selectbox('Car Body', 
            ['sedan', 'hatchback', 'wagon', 'hardtop', 'convertible']),
        'drivewheel': st.sidebar.selectbox('Drive Wheel', 
            ['fwd', 'rwd', '4wd']),
        'enginelocation': st.sidebar.selectbox('Engine Location', 
            ['front', 'rear']),
        'wheelbase': st.sidebar.slider('Wheelbase', 86.6, 120.9, 98.8),
        'carlength': st.sidebar.slider('Car Length', 141.1, 208.1, 174.6),
        'carwidth': st.sidebar.slider('Car Width', 60.3, 72.3, 66.3),
        'carheight': st.sidebar.slider('Car Height', 47.8, 59.8, 53.8),
        'curbweight': st.sidebar.slider('Curb Weight', 1488, 4066, 2777),
        'enginetype': st.sidebar.selectbox('Engine Type', 
            ['dohc', 'ohcv', 'ohc', 'l', 'rotor', 'ohcf', 'dohcv']),
        'cylindernumber': st.sidebar.selectbox('Number of Cylinders', 
            ['four', 'six', 'five', 'eight', 'two', 'twelve', 'three']),
        'enginesize': st.sidebar.slider('Engine Size', 61, 326, 126),
        'fuelsystem': st.sidebar.selectbox('Fuel System', 
            ['mpfi', '2bbl', 'mfi', '1bbl', '4bbl', 'idi', 'spfi', 'spdi']),
        'boreratio': st.sidebar.slider('Bore Ratio', 2.54, 3.94, 3.24),
        'stroke': st.sidebar.slider('Stroke', 2.07, 4.17, 3.12),
        'compressionratio': st.sidebar.slider('Compression Ratio', 7.0, 23.0, 15.0),
        'horsepower': st.sidebar.slider('Horsepower', 48, 288, 168),
        'peakrpm': st.sidebar.slider('Peak RPM', 4150, 6600, 5375),
        'citympg': st.sidebar.slider('City MPG', 13, 49, 31),
        'highwaympg': st.sidebar.slider('Highway MPG', 16, 54, 35),
    }
    
    # Make predictions
    if st.button('Predict Price'):
        # Preprocess input data
        features_df = preprocess_input(input_data)
        
        # Scale features using the same scaler used during training
        features_scaled = scaler.transform(features_df)
        
        # Make predictions
        lr_pred = lr_model.predict(features_scaled)[0]
        dt_pred = dt_model.predict(features_scaled)[0]
        rf_pred = rf_model.predict(features_scaled)[0]
        
        # Display predictions
        st.write('### Price Predictions')
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write('Linear Regression:')
            st.write(f'${lr_pred:,.2f}')
        
        with col2:
            st.write('Decision Tree:')
            st.write(f'${dt_pred:,.2f}')
        
        with col3:
            st.write('Random Forest:')
            st.write(f'${rf_pred:,.2f}')
    
    # Display model performance metrics
    st.write('### Model Performance Metrics')
    performance_data = pd.read_csv('model_performance.csv')
    st.dataframe(performance_data)
    
    # Display visualizations
    st.write('### Visualizations')
    
    # Show correlation heatmap
    st.write('#### Feature Correlation Heatmap')
    img = plt.imread('visualizations/correlation_heatmap.png')
    st.image(img)
    
    # Show model comparison plots
    st.write('#### Model Comparison')
    metrics = ['r2_score', 'rmse', 'mae', 'mpe']
    for metric in metrics:
        img = plt.imread(f'visualizations/{metric}_comparison.png')
        st.image(img)

if __name__ == '__main__':
    main()
