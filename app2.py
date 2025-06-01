import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')  # Suppress non-critical warnings
import matplotlib.animation as animation
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
from plotly.subplots import make_subplots
import time
from streamlit_lottie import st_lottie
import requests

st.set_page_config(page_title="Prediction Dashboard", layout="wide")




# Custom CSS for Bona Nova font with higher specificity
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bona+Nova&display=swap');     
    /* Target Streamlit's root container */
    .stApp, .stApp * {
        font-family: 'Bona Nova', serif !important;
        font-weight: 400 !important;
        font-style: normal !important;
    }
    
    /* Target specific Streamlit elements */
    h1, h2, h3, h4, h5, h6, .stMarkdown, .stText, .stSelectbox, .stCheckbox, .stDataFrame, .stTable {
        font-family: 'Bona Nova', serif !important;
        font-weight: 400 !important;
        font-style: normal !important;
    }
    
    /* Ensure sidebar uses the font */
    [data-testid="stSidebar"], [data-testid="stSidebar"] * {
        font-family: 'Bona Nova', serif !important;
        font-weight: 400 !important;
        font-style: normal !important;
    }
    
    /* Debug font loading */
    @font-face {
        font-family: 'Bona Nova';
        font-weight: 400 !important;
        font-style: normal !important;
        src: url(@import url('https://fonts.googleapis.com/css2?family=Bona+Nova&display=swap');
    }
    </style>
    <div style="font-family: 'Bona Nova', serif; 
     font-weight: 400 !important;
        font-style: normal !important;display: none;">Font load test</div>
    """,
    unsafe_allow_html=True
)

# Sidebar for user interaction
st.sidebar.markdown(
    """
    <style>
    /* Style for Navigation header background */
    .navigation-header {
        font-size: 1.5em;
    border-bottom: 2px solid #cccc;
    font-weight: bold;
    background-color: #081c15;
    margin-bottom: 2px;
    /* border-radius: 5px; */
    width: 118%;
    padding: 10px;
    position: absolute;
    height: 128px;
    top: -90px;
    left: -25px;
    }
    
    .navigation-header p{
    color: white;
    position: absolute;
    top: 60px;
    left: 26px;
    }
    </style>
    <div class="navigation-header"><p>Ecomove</p></div>
    
    """,
    unsafe_allow_html=True
)



# Set page config
# Set Matplotlib style to ensure light backgrounds
plt.style.use('default')  # Reset to default Matplotlib style
sns.set_style("whitegrid")  # White background with grid
# Sidebar navigation
#st.sidebar.title("Navigation")
# Sidebar navigation
#st.sidebar.title("Navigation")
page = st.sidebar.radio(" ", [
    "Home", 
    "Motor Cars", 
    "Battery Cars", 
    "AQI Category" 
])
data = pd.read_csv('dataset/ts.csv')

# Home Page
if page == "Home":
    st.title("Prediction Dashboard")
    st.write("Explore predictions and visualizations for motor cars, battery electric vehicles, and AQI classification")
    #st.markdown("[Visit Front-End Website](../frontend/index.html)")
    st.markdown("Ecomove is your gateway to understanding the environmental footprint of the vehicles that power our world. By harnessing real-time data analysis, Ecomove compares the impact of Petrol, Diesel, and Electric vehicles through key pollutants like CO₂, NOx, and more. From buses to motorcycles, our interactive platform reveals which fuel types harm our planet most and forecasts future trends to guide sustainable choices. Explore stunning visualizations, dive into predictive insights, and join us in steering toward a greener future with Ecomove! \n [Visit Front-End Website](../frontend/index.html)")
    st.write("Use the sidebar to navigate to different sections.")


# Motor Cars Regression
elif page == "Battery Cars":
    st.title("CO₂ Production from Battery Electric Vehicles (BEV)")
    # Graph 1: Actual vs Predicted CO₂ Emissions (2010–2020)
    fig1 = go.Figure()

    

    X = data[['year', 'BEV_sales']]
    y = data['bev_kt']
    
    # Log-transform bev_sales to reduce skewness
    X['bev_sales_log'] = np.log1p(X['BEV_sales'])

    
    # Create year_offset feature
    X['year_offset'] = X['year'] - 2010
    # Features to use: year_offset and bev_sales_log
    X_final = X[['year_offset', 'bev_sales_log']]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_final)
    
    # Save preprocessed data
    preprocessed_data = {'X_scaled': X_scaled, 'y': y, 'scaler': scaler, 'X_final': X_final}
    X_scaled = preprocessed_data['X_scaled']
    y = preprocessed_data['y']
    scaler = preprocessed_data['scaler']
    X_final = preprocessed_data['X_final']
    poly = PolynomialFeatures(degree=3, include_bias=False)
    X_final_poly = poly.fit_transform(X_final)
    X_scaled_poly = scaler.fit_transform(X_final_poly)
    
    param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
    ridge_model_poly = GridSearchCV(Ridge(), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    ridge_model_poly.fit(X_scaled_poly, y)
    best_ridge_poly = ridge_model_poly.best_estimator_
    y_pred_train_poly = np.maximum(best_ridge_poly.predict(X_scaled_poly), 0)
    
    actual_pred_df = pd.DataFrame({
    'Year': data['year'],
    'BEV Sales': data['BEV_sales'],
    'Actual battery production': y,
    'Predicted battery production': y_pred_train_poly
    })







    # Graph 2: Predicted CO₂ Emissions (2021–2030)
    fig2 = go.Figure()

    #PREDICTION FROM 2021-2030
    # Create future data for 2021–2030
    years_future = np.arange(2021, 2031)
    # Assume 10% exponential growth for BEV Sales from 2,000,000 (2020)
    bev_sales_2020 = 2000000
    bev_sales_future = bev_sales_2020 * (1 + 0.1) ** (years_future - 2020)

    # Replace with specific BEV Sales if available, e.g.:
    # bev_sales_future = [2200000, 2500000, 2900000, 3400000, 4000000, 4700000, 5500000, 6400000, 7400000, 8500000]

    # Prepare future data
    future_data = pd.DataFrame({
        'year': years_future,
        'BEV_sales': bev_sales_future
    })
    future_data['year_offset'] = future_data['year'] - 2010
    future_data['bev_sales_log'] = np.log1p(future_data['BEV_sales'])
    future_data_poly = poly.transform(future_data[['year_offset', 'bev_sales_log']])
    future_data_scaled_poly = scaler.transform(future_data_poly)

    # Predict bev_kt for 2021–2030 and clip to avoid negatives
    y_pred_future_poly = np.maximum(best_ridge_poly.predict(future_data_scaled_poly), 0)

    # Create table of predictions for 2021–2030
    predictions_df = pd.DataFrame({
        'Year': years_future,
        'BEV Sales': bev_sales_future,
        'Predicted battery production': y_pred_future_poly
    })





    # Add scatter for actual values (2010–2020)
    fig2.add_trace(
        go.Scatter(
            x=data['year'],
            y=y,
            mode='markers',
            name='Actual Battery Production',
            marker=dict(color='green', size=10, opacity=0.8),
            hovertemplate='Year: %{x}<br>CO₂ (kt): %{y:.2f}<extra></extra>'
        )
    )

    # Add line for predicted values (2021–2030)
    fig2.add_trace(
        go.Scatter(
            x=years_future,
            y=y_pred_future_poly,
            mode='lines+markers',
            name='Predicted Battery Production',
            line=dict(color='blue', width=2),
            marker=dict(size=8),
            hovertemplate='Year: %{x}<br>CO₂ (kt): %{y:.2f}<extra></extra>'
        )
    )

    # Add transition line at 2020
    fig2.add_vline(x=2020, line=dict(color='gray', dash='dash'), annotation_text='2020 (Transition)')

    fig2.update_layout(
        title='CO₂ Produced from Battery: Actual (2010–2020) and Predicted (2021–2030)',
        xaxis_title='Year',
        yaxis_title='CO₂ Produced from Battery (megatonnes)',
        hovermode='x unified',
        showlegend=True,
        template='plotly_white',
        xaxis=dict(gridcolor='rgba(200,200,200,0.3)'),
        yaxis=dict(gridcolor='rgba(200,200,200,0.3)')
    )

    st.plotly_chart(fig2, use_container_width=True)

    # Graph 3: ARIMA Forecast (2021–2035)
    fig3 = go.Figure()

    # Add line for historical data
    fig3.add_trace(
        go.Scatter(
            x=data.index,
            y=data['bev_kt'],
            mode='lines+markers',
            name='Historical Battery Production',
            line=dict(color='blue', width=2),
            marker=dict(size=8),
            hovertemplate='Year: %{x|%Y}<br>CO₂ (kt): %{y:.2f}<extra></extra>'
        )
    )
    
    # ARIMA forecast
    # Set 'year' as the index for ARIMA (optional, but helps with time series)
    data['year'] = pd.to_datetime(data['year'], format='%Y')  # Convert year to datetime
    data.set_index('year', inplace=True)

    # Fit ARIMA model (p=1, d=1, q=1 as a starting point)
    model = ARIMA(data['bev_kt'], order=(1, 1, 1))
    model_fit = model.fit()

    # Summary of the model
    #print(model_fit.summary())

    # Forecast next 15 years (2021-2035)
    forecast_steps = 15
    forecast = model_fit.forecast(steps=forecast_steps)

    # Create a date range for the forecast (assuming last year in data is 2020)
    last_year = data.index[-1].year
    forecast_years = pd.date_range(start=f'{last_year + 1}-01-01', periods=forecast_steps, freq='YS')
    forecast_series = pd.Series(forecast, index=forecast_years)

   
    # Add line for forecast
    # Create Plotly figure
    # Combine for setting accurate x-axis range
    combined_years = list(data.index.year) + list(forecast_series.index.year)

    # Plot
    fig3 = go.Figure()

    # Historical trace
    fig3.add_trace(
        go.Scatter(
            x=data.index.year,
            y=data['bev_kt'],
            mode='lines+markers',
            name='Historical',
            line=dict(color='royalblue', width=3),
            marker=dict(size=6),
            hovertemplate='Year: %{x}<br>CO₂: %{y:.2f} kt<extra></extra>'
        )
    )

    # Forecast trace
    fig3.add_trace(
        go.Scatter(
            x=forecast_series.index.year,
            y=forecast_series,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='orange', width=3, dash='dash'),
            marker=dict(size=6),
            hovertemplate='Year: %{x}<br>CO₂: %{y:.2f} kt<extra></extra>'
        )
    )

    # Final layout
    fig3.update_layout(
        title='CO₂ Produced from Battery Forecast (2021–2035)',
        xaxis_title='Year',
        yaxis_title='CO₂ produced from Battery (megatonnes)',
        hovermode='x unified',
        template='plotly_white',
        xaxis=dict(
            tickmode='linear',
            dtick=1,
            range=[min(combined_years) - 1, max(combined_years) + 1],  # clean range
            gridcolor='rgba(200,200,200,0.3)'
        ),
        yaxis=dict(
            gridcolor='rgba(200,200,200,0.3)',
            range=[data['bev_kt'].min() - 10, forecast_series.max() + 10]
        )
    )

    # In Streamlit

    st.plotly_chart(fig3, use_container_width=True)


elif page == "Motor Cars":
    st.title("CO₂ Production from Motor Vehicles") 
    
    # 1. Load Data
    X = data[['year', 'total_motor_cars_production']]
    y = data['motor_mt']

    # 2. Add Polynomial Features
    poly = PolynomialFeatures(degree=2, include_bias=False)  # Degree=2 for quadratic terms
    X_poly = poly.fit_transform(X)
    poly_feature_names = poly.get_feature_names_out(X.columns)
    X_poly_df = pd.DataFrame(X_poly, columns=poly_feature_names)

    # 3. Scale Features
    scaler = StandardScaler()
    X_poly_scaled = scaler.fit_transform(X_poly_df)
    X_poly_scaled_df = pd.DataFrame(X_poly_scaled, columns=poly_feature_names)

    # 4. Train Ridge Model
    ridge_model = Ridge(alpha=0.1, random_state=42)
    ridge_model.fit(X_poly_scaled_df, y)
    predictions = ridge_model.predict(X_poly_scaled_df)

    

    # 6. Create Table for Actual vs Predicted
    results_2010_2020 = pd.DataFrame({
        'Year': data['year'],
        'Total Motor Cars Production': data['total_motor_cars_production'],
        'Actual motor_mt': y,
        'Predicted motor_mt': predictions
    })
# 1. Load and Preprocess Data

    # 2. Polynomial Features and Scaling
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    poly_feature_names = poly.get_feature_names_out(X.columns)
    X_poly_df = pd.DataFrame(X_poly, columns=poly_feature_names)

    scaler = StandardScaler()
    X_poly_scaled = scaler.fit_transform(X_poly_df)
    X_poly_scaled_df = pd.DataFrame(X_poly_scaled, columns=poly_feature_names)

    # 3. Train Ridge Model
    ridge_model = Ridge(alpha=0.1, random_state=42)
    ridge_model.fit(X_poly_scaled_df, y)
    predictions = ridge_model.predict(X_poly_scaled_df)

    # 4. Forecast Total Motor Cars Production with Linear Regression
    lr = LinearRegression()
    years = X['year'].values.reshape(-1, 1)
    production = X['total_motor_cars_production'].values
    lr.fit(years, production)

    # Generate future years (2021–2030)
    future_years = np.arange(2021, 2031).reshape(-1, 1)
    future_production = lr.predict(future_years)

    # Create future data DataFrame
    future_data = pd.DataFrame({
        'year': future_years.flatten(),
        'total_motor_cars_production': future_production
    })

    # 5. Apply Polynomial Features and Scale for Future Data
    future_poly = poly.transform(future_data)
    future_poly_df = pd.DataFrame(future_poly, columns=poly_feature_names)
    future_poly_scaled = scaler.transform(future_poly_df)
    future_poly_scaled_df = pd.DataFrame(future_poly_scaled, columns=poly_feature_names)

    # 6. Predict motor_kt for 2021–2030
    future_predictions = ridge_model.predict(future_poly_scaled_df)

    # 7. Table for 2021–2030 Predictions
    future_results = pd.DataFrame({
        'Year': future_data['year'],
        'Total Motor Cars Production': future_data['total_motor_cars_production'],
        'Predicted motor_mt': future_predictions
    })
    
    
    fig2 = go.Figure()

    # Add scatter for actual values (2010–2020)
    fig2.add_trace(
        go.Scatter(
            x=data['year'],
            y=y,
            mode='markers',
            name='Actual Motor Production',
            marker=dict(color='green', size=10, opacity=0.8),
            hovertemplate='Year: %{x}<br>CO₂ (kt): %{y:.2f}<extra></extra>'
        )
    )

    # Add line for predicted values (2021–2030)
    fig2.add_trace(
        go.Scatter(
            x=future_data['year'],
            y=future_predictions,
            mode='lines+markers',
            name='Predicted Motor Production',
            line=dict(color='blue', width=2),
            marker=dict(size=8),
            hovertemplate='Year: %{x}<br>CO₂ (kt): %{y:.2f}<extra></extra>'
        )
    )

    # Add transition line at 2020
    fig2.add_vline(x=2020, line=dict(color='gray', dash='dash'), annotation_text='2020 (Transition)')

    fig2.update_layout(
        title='CO₂ Produced from Motor: Actual (2010–2020) and Predicted (2021–2030)',
        xaxis_title='Year',
        yaxis_title='CO₂ Produced from Motor (megatonnes)',
        hovermode='x unified',
        showlegend=True,
        template='plotly_white',
        xaxis=dict(gridcolor='rgba(200,200,200,0.3)'),
        yaxis=dict(gridcolor='rgba(200,200,200,0.3)')
    )

    st.plotly_chart(fig2, use_container_width=True)
    
    
    
    
    #time series
    # Set 'year' as the index for ARIMA (optional, but helps with time series)
    data['year'] = pd.to_datetime(data['year'], format='%Y')  # Convert year to datetime
    data.set_index('year', inplace=True)

    # Fit ARIMA model (p=1, d=1, q=1 as a starting point)
    model = ARIMA(data['motor_mt'], order=(1, 1, 1))
    model_fit = model.fit()

    # Summary of the model
    print(model_fit.summary())

    # Forecast next 15 years (2021-2035)
    forecast_steps = 15
    forecast = model_fit.forecast(steps=forecast_steps)

    # Create a date range for the forecast (assuming last year in data is 2020)
    last_year = data.index[-1].year
    forecast_years = pd.date_range(start=f'{last_year + 1}-01-01', periods=forecast_steps, freq='YS')
    forecast_series = pd.Series(forecast, index=forecast_years)

   
    
    
    # Combine years for x-axis range
    combined_years = list(data.index.year) + list(forecast_series.index.year)

    # Create figure
    fig3 = go.Figure()

    # Historical data trace
    fig3.add_trace(go.Scatter(
        x=data.index.year,
        y=data['motor_mt'],
        mode='lines+markers',
        name='Historical',
        line=dict(color='royalblue', width=3),
        marker=dict(size=6),
        hovertemplate='Year: %{x}<br>CO₂: %{y:.2f} Mt<extra></extra>'
    ))

    # Forecast trace
    fig3.add_trace(go.Scatter(
        x=forecast_series.index.year,
        y=forecast_series,
        mode='lines+markers',
        name='Forecast',
        line=dict(color='orange', width=3, dash='dash'),
        marker=dict(size=6),
        hovertemplate='Year: %{x}<br>CO₂: %{y:.2f} Mt<extra></extra>'
    ))

    # Update layout
    fig3.update_layout(
        title='CO₂ Emissions from Motor Forecast (2021–2035)',
        xaxis_title='Year',
        yaxis_title='CO₂ Emission from Motor (Mt)',
        hovermode='x unified',
        template='plotly_white',
        xaxis=dict(
            tickmode='linear',
            dtick=1,
            range=[min(combined_years) - 1, max(combined_years) + 1],
            gridcolor='rgba(200,200,200,0.3)'
        ),
        yaxis=dict(
        range=[
            min(data['motor_mt'].min(), forecast_series.min()) - 10,
            max(data['motor_mt'].max(), forecast_series.max()) + 10
        ],
        gridcolor='rgba(200,200,200,0.3)'
    )
    )
    # Display in Streamlit
    st.plotly_chart(fig3, use_container_width=True)
    
    

elif page == "AQI Category":
    st.title("AQI Category")

    # --- AQI Calculation Functions ---
    # --- AQI Calculation Functions ---
    def nox_gkm_to_ugm3(nox_gkm, speed, fuel_type='petrol'):
        distance_km = speed
        emission_g = nox_gkm * distance_km
        emission_ug = emission_g * 1e6
        air_volume_m3 = 1000000 if fuel_type == 'diesel' else 900000  
        nox_ugm3 = emission_ug / air_volume_m3
        return nox_ugm3

    def calculate_nox_aqi(nox_emissions):
        if nox_emissions <= 40:
            return (50 / 40) * nox_emissions
        elif nox_emissions <= 80:
            return ((100 - 51) / (80 - 41)) * (nox_emissions - 41) + 51
        elif nox_emissions <= 180:
            return ((200 - 101) / (180 - 81)) * (nox_emissions - 81) + 101
        elif nox_emissions <= 280:
            return ((300 - 201) / (280 - 181)) * (nox_emissions - 181) + 201
        elif nox_emissions <= 400:
            return ((400 - 301) / (400 - 281)) * (nox_emissions - 281) + 301
        else:
            return ((nox_emissions - 401) / (800 - 401)) * (nox_emissions - 401) + 401

    def get_aqi_category(aqi):
        if aqi <= 50:
            return "Good"
        elif aqi <= 100:
            return "Moderate"
        elif aqi <= 150:
            return "Unhealthy for Sensitive Groups"
        elif aqi <= 200:
            return "Unhealthy"
        elif aqi <= 300:
            return "Very Unhealthy"
        elif aqi <= 500:
            return "Hazardous"
        else:
            return "Hazardous"

    def predict_nox_emissions(user_input, model, scaler, features, numerical_cols):
        input_df = pd.DataFrame([user_input], columns=['Engine Size', 'Age of Vehicle', 'Temperature',
                                                    'Wind Speed', 'Speed', 'Traffic Conditions', 'Road Type'])
        input_df = pd.get_dummies(input_df, columns=['Traffic Conditions', 'Road Type'], drop_first=True)
        for col in features:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[features]
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
        nox_emission = model.predict(input_df)[0]
        return max(0, nox_emission)


   # --- Streamlit App ---
    st.title("NOx Emission and AQI Prediction for Vehicles")

    # Sidebar for input parameters
    with st.sidebar:
        st.header("Input Vehicle Parameters")
        st.markdown("**Maximum Allowable Input Values:**")
        st.write("- **Engine Size**: 6.0 liters")
        st.write("- **Age of Vehicle**: 30 years")
        st.write("- **Temperature**: 40°C")
        st.write("- **Vehicle Speed**: 120 km/h")
        st.write("- **Wind Speed**: 30 km/h")
        st.markdown("---")
        fuel_type = st.selectbox("Fuel Type", ["Petrol", "Electric", "Diesel"])
        engine_size = st.number_input("Engine Size (liters)", min_value=0.0, max_value=6.0, value=2.0, step=0.1)
        age = st.number_input("Age of Vehicle (years)", min_value=0, max_value=30, value=5)
        temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=40.0, value=25.0, step=1.0)
        speed = st.number_input("Vehicle Speed (km/h)", min_value=10.0, max_value=120.0, value=50.0, step=5.0)
        wind_speed = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=30.0, value=10.0, step=1.0)
        traffic_condition = st.selectbox("Traffic Condition", ["Free flow", "Moderate", "Heavy"])
        road_type = st.selectbox("Road Type", ["City", "Rural", "Highway"])

    # Load models and scalers
    models = {}
    scalers = {}
    try:
        models['Petrol'] = joblib.load('aqi/gradient_boosting_model.pkl')
        scalers['Petrol'] = joblib.load('aqi/scaler.pkl')
        models['Electric'] = joblib.load('aqi/gradient_boosting_electric_model.pkl')
        scalers['Electric'] = joblib.load('aqi/scaler_electric.pkl')
        models['Diesel'] = joblib.load('aqi/svr_diesel_model.pkl')
        scalers['Diesel'] = joblib.load('aqi/scaler_diesel.pkl')
    except FileNotFoundError:
        st.error("One or more model/scaler files not found. Ensure all .pkl files are in the 'aqi' directory.")
        st.stop()

    # Define features (consistent across all fuel types)
    features = [
        'Engine Size', 'Age of Vehicle', 'Temperature', 'Wind Speed', 'Speed','Traffic Conditions_Heavy',
        'Traffic Conditions_Moderate',
        'Road Type_Highway', 'Road Type_Rural'
    ]
    numerical_cols = ['Engine Size', 'Age of Vehicle', 'Temperature', 'Wind Speed', 'Speed']

    # Initialize session state
    if 'results' not in st.session_state:
        st.session_state.results = {ft: {'nox_emission': None, 'nox_aqi': None, 'aqi_category': None} for ft in ['Petrol', 'Electric', 'Diesel']}

    # Compute NOx and AQI for selected fuel type
    if st.button("Get Result"):
        if engine_size < 0.5 and fuel_type != 'Electric':
            st.warning("Engine Size is too small (less than 0.5 liters). Assuming NOx Emission is 0.")
            nox_emission_gkm = 0.0
        else:
            user_input = {
                'Engine Size': engine_size,
                'Age of Vehicle': age,
                'Temperature': temperature,
                'Wind Speed': wind_speed,
                'Speed': speed,
                'Traffic Conditions': traffic_condition,
                'Road Type': road_type
            }
            # Compute for selected fuel type
            nox_emission_gkm = predict_nox_emissions(user_input, models[fuel_type], scalers[fuel_type], features, numerical_cols)
            st.session_state.results[fuel_type]['nox_emission'] = nox_emission_gkm
            st.write(f"**Debug: Predicted NOx Emission ({fuel_type})**: {nox_emission_gkm:.4f} g/km")

        st.subheader(f"Prediction Result ({fuel_type})")
        st.write(f"**NOx Emission**: {nox_emission_gkm:.4f} g/km")

        if (engine_size < 0.0 or engine_size > 6.0 or age < 0 or age > 30 or
            temperature < -10 or temperature > 40 or speed < 10 or speed > 120 or
            wind_speed < 0 or wind_speed > 30):
            st.warning("Input values are outside typical ranges. Predictions may be less reliable.")

    # Compute AQI for all fuel types
    if st.button("Get AQI"):
        comparison_data = []
        for ft in ['Petrol', 'Electric', 'Diesel']:
            if st.session_state.results[ft]['nox_emission'] is not None:
                nox_emission_ugm3 = nox_gkm_to_ugm3(st.session_state.results[ft]['nox_emission'], speed, fuel_type=ft)
                nox_aqi = calculate_nox_aqi(nox_emission_ugm3)
                aqi_category = get_aqi_category(nox_aqi)
                st.session_state.results[ft]['nox_aqi'] = nox_aqi
                st.session_state.results[ft]['aqi_category'] = aqi_category

                st.subheader(f"AQI Result ({ft})")
                st.write(f"**Debug: NOx Concentration**: {nox_emission_ugm3:.2f} µg/m³")
                st.write(f"**NOx Emission (converted)**: {nox_emission_ugm3:.2f} µg/m³")
                st.write(f"**NOx AQI**: {nox_aqi:.1f}")
                st.write(f"**AQI Category**: {aqi_category}")

                # Individual bar chart
                results = pd.DataFrame({
                    'Metric': ['NOx Emission (µg/m³)', 'NOx AQI'],
                    'Value': [nox_emission_ugm3, nox_aqi]
                })
                fig = px.bar(results, x='Metric', y='Value', title=f'NOx Emission and AQI ({ft})',
                            color='Metric', text='Value', height=400)
                fig.update_traces(texttemplate='%{text:.2f}', textposition='auto')
                st.plotly_chart(fig)

                comparison_data.append({
                    'Fuel Type': ft,
                    'NOx AQI': nox_aqi,
                    'AQI Category': aqi_category,
                    'NOx Emission (µg/m³)': nox_emission_ugm3
                })

        # Comparison bar chart
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            fig = px.bar(comparison_df, x='Fuel Type', y='NOx AQI',
                        color='AQI Category', title='NOx AQI Comparison Across Fuel Types',
                        text='NOx AQI', height=500)
            fig.update_traces(texttemplate='%{text:.1f}', textposition='auto')
            st.plotly_chart(fig)

            worst_fuel = comparison_df.loc[comparison_df['NOx AQI'].idxmax(), 'Fuel Type']
            st.write(f"**Conclusion**: {worst_fuel} has the highest NOx AQI ({comparison_df['NOx AQI'].max():.1f}), indicating it is the most harmful to the environment.")
        else:
            st.info("Please click 'Get Result' for at least one fuel type to compute AQI.") 
            


