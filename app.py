import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin

# Set page configuration
st.set_page_config(
    page_title="Hotel Cancellation Predictor",
    page_icon="🏨",
    layout="wide"
)

# Title and description
st.title("Hotel Cancellation Prediction Dashboard")
st.markdown("""
This dashboard predicts the likelihood of a hotel booking cancellation and shows
the factors that influenced this prediction using SHAP values.
""")

# Reuse your transformer class (required for loading the model)
class BookingDataTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Work on a copy so we do not modify the original DataFrame
        df = X.copy()

        # --- Filter out problematic rows ---
        # Remove rows where adults, children, and babies are all zero
        mask = (df['adults'] == 0) & (df['children'] == 0) & (df['babies'] == 0)
        df = df.loc[~mask]

        # Remove rows where adults is zero but children or babies are positive
        mask = (df['adults'] == 0) & ((df['children'] > 0) | (df['babies'] > 0))
        df = df.loc[~mask]

        # --- Fill missing values and create features ---
        # Fill missing 'agent' with max(agent)+1
        df['agent'] = df['agent'].fillna(df['agent'].max() + 1)

        # Fill missing 'children' with 0 (so that is_family can be computed)
        df['children'] = df['children'].fillna(0)
        df['is_family'] = (df['children'] + df['babies']) > 0

        # Drop columns not needed
        df = df.drop(['company', 'children', 'babies'], axis=1)

        # --- Create country-based features ---
        european_countries = [
            'GBR', 'FRA', 'ESP', 'DEU', 'ITA', 'IRL', 'BEL', 'NLD', 'CHE', 'AUT', 'SWE', 'POL', 
            'RUS', 'NOR', 'ROU', 'FIN', 'DNK', 'LUX', 'TUR', 'HUN', 'CZE', 'GRC', 'SRB', 'HRV', 
            'EST', 'LTU', 'BGR', 'UKR', 'SVK', 'ISL', 'SVN', 'LVA', 'CYP', 'MNE', 'AND', 'MLT', 
            'GIB', 'BIH', 'ALB', 'MKD', 'LIE', 'SMR', 'FRO', 'MCO'
        ]
        
        # Define the conditions.
        conditions = [
            df['country'] == 'PRT',                 # Condition for Portugal.
            df['country'].isin(european_countries)   # Condition for other European countries.
        ]
        
        # Define the corresponding choices for each condition.
        choices = ['Portugal', 'European']
        
        # Create the new column, with 'Rest of the world' as the default.
        df['country_of_origin'] = np.select(conditions, choices, default='Rest of the world')
        
        # Optionally, drop the original 'country' column.
        df = df.drop('country', axis=1)

        # --- Process room types ---
        df['room_status'] = (df['reserved_room_type'] == df['assigned_room_type']).astype(int)
        df = df.drop(['reserved_room_type', 'assigned_room_type'], axis=1)

        # --- Create stay-related features ---
        df['total_stay_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
        df = df.drop(['stays_in_weekend_nights', 'stays_in_week_nights'], axis=1)

        # --- Create cancellation and request indicators ---
        df['cancellation_risk'] = (df['previous_cancellations'] > 0).astype(int)
        df = df.drop('previous_cancellations', axis=1)

        df['special_request_indicator'] = (df['total_of_special_requests'] > 0).astype(int)
        df = df.drop('total_of_special_requests', axis=1)

        df['waiting_list_indicator'] = (df['days_in_waiting_list'] > 0).astype(int)
        df = df.drop('days_in_waiting_list', axis=1)

        df['booking_changes_indicator'] = (df['booking_changes'] > 0).astype(int)
        df = df.drop('booking_changes', axis=1)

        df['required_car_spaces_indicator'] = (df['required_car_parking_spaces'] > 0).astype(int)
        df = df.drop('required_car_parking_spaces', axis=1)

        # --- Process arrival date information ---
        df['arrival_date'] = pd.to_datetime(
            df['arrival_date_day_of_month'].astype(str) + ' ' +
            df['arrival_date_month'] + ' ' +
            df['arrival_date_year'].astype(str),
            errors='coerce'
        )
        df['arrival_day_name'] = df['arrival_date'].dt.day_name()
        seasons = {
            'Winter': ['December', 'January', 'February'],
            'Spring': ['March', 'April', 'May'],
            'Summer': ['June', 'July', 'August'],
            'Fall': ['September', 'October', 'November']
        }
        def get_season(month):
            for season, months in seasons.items():
                if month in months:
                    return season
            return 'Unknown'
        df['season'] = df['arrival_date_month'].apply(get_season)
        df = df.drop(['arrival_date_day_of_month', 'arrival_date_month', 
                    'arrival_date_week_number', 'arrival_date'], axis=1)

        return df

# Load trained model and preprocessing pipeline
@st.cache_resource
def load_model():
    model = joblib.load('dashboard/models/best_model.pkl')
    preprocessing_pipeline = joblib.load('dashboard/models/preprocessing_pipeline.pkl')
    return model, preprocessing_pipeline

try:
    model, preprocessing_pipeline = load_model()
    st.success("Model and preprocessing pipeline loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Create input form
st.subheader("Enter Booking Information")

# Create two columns for form layout
col1, col2 = st.columns(2)

with col1:
    # Numerical inputs
    adults = st.number_input("Number of Adults", min_value=1, max_value=10, value=2)
    adr = st.number_input("Average Daily Rate (EUR)", min_value=0.0, max_value=1000.0, value=100.0)
    lead_time = st.number_input("Lead Time (days)", min_value=0, max_value=365, value=30)
    
    # Stay details
    weekend_nights = st.number_input("Stays in Weekend Nights", min_value=0, max_value=10, value=1)
    week_nights = st.number_input("Stays in Week Nights", min_value=0, max_value=30, value=3)
    
    # More booking details
    is_repeated_guest = st.selectbox("Is Repeated Guest", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    previous_bookings_not_canceled = st.number_input("Previous Bookings Not Canceled", min_value=0, max_value=50, value=0)
    
    # Room details
    reserved_room_type = st.selectbox("Reserved Room Type", options=["A", "B", "C", "D", "E", "F", "G", "H"])
    assigned_room_type = st.selectbox("Assigned Room Type", options=["A", "B", "C", "D", "E", "F", "G", "H"])

with col2:
    # Categorical inputs
    market_segment = st.selectbox("Market Segment", options=["Direct", "Corporate", "Online TA", "Offline TA/TO", "Complementary", "Groups", "Aviation"])
    distribution_channel = st.selectbox("Distribution Channel", options=["Direct", "Corporate", "TA/TO", "GDS", "Undefined"])
    country = st.selectbox("Country", options=["PRT", "GBR", "FRA", "ESP", "DEU", "ITA", "USA", "Other"])
    if country == "Other":
        country = st.text_input("Specify Country Code (3 letters)")
    
    # More booking details
    deposit_type = st.selectbox("Deposit Type", options=["No Deposit", "Non Refund", "Refundable"])
    customer_type = st.selectbox("Customer Type", options=["Transient", "Transient-Party", "Contract", "Group"])
    meal = st.selectbox("Meal", options=["BB", "FB", "HB", "SC", "Undefined"])
    
    # Additional details
    children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
    babies = st.number_input("Number of Babies", min_value=0, max_value=10, value=0)
    
    # Date inputs
    arrival_date_year = st.selectbox("Arrival Year", options=[2015, 2016, 2017])
    arrival_date_month = st.selectbox("Arrival Month", options=[
        "January", "February", "March", "April", "May", "June", 
        "July", "August", "September", "October", "November", "December"
    ])
    arrival_date_day = st.number_input("Arrival Day", min_value=1, max_value=31, value=15)

# Additional inputs
col1, col2, col3 = st.columns(3)
with col1:
    agent = st.number_input("Agent ID", min_value=0, max_value=535, value=0)
    company = st.number_input("Company ID", min_value=0, max_value=500, value=0)
    
with col2:
    previous_cancellations = st.number_input("Previous Cancellations", min_value=0, max_value=50, value=0)
    booking_changes = st.number_input("Booking Changes", min_value=0, max_value=20, value=0)
    days_in_waiting_list = st.number_input("Days in Waiting List", min_value=0, max_value=365, value=0)

with col3:    
    required_car_parking_spaces = st.number_input("Required Car Parking Spaces", min_value=0, max_value=8, value=0)
    total_of_special_requests = st.number_input("Total of Special Requests", min_value=0, max_value=5, value=0)
    arrival_date_week_number = st.number_input("Arrival Date Week Number", min_value=1, max_value=53, value=25)

# Create a button to trigger prediction
predict_button = st.button("Predict Cancellation Probability", type="primary")

if predict_button:
    # Create a single row dataframe with the input values
    input_data = pd.DataFrame({
        'hotel': ['Resort Hotel'],
        'adults': [adults],
        'children': [children],
        'babies': [babies],
        'meal': [meal],
        'country': [country],
        'market_segment': [market_segment],
        'distribution_channel': [distribution_channel],
        'is_repeated_guest': [is_repeated_guest],
        'previous_cancellations': [previous_cancellations],
        'previous_bookings_not_canceled': [previous_bookings_not_canceled],
        'reserved_room_type': [reserved_room_type],
        'assigned_room_type': [assigned_room_type],
        'booking_changes': [booking_changes],
        'deposit_type': [deposit_type],
        'agent': [agent],
        'company': [company],
        'days_in_waiting_list': [days_in_waiting_list],
        'customer_type': [customer_type],
        'adr': [adr],
        'required_car_parking_spaces': [required_car_parking_spaces],
        'total_of_special_requests': [total_of_special_requests],
        'arrival_date_year': [arrival_date_year],
        'arrival_date_month': [arrival_date_month],
        'arrival_date_week_number': [arrival_date_week_number],
        'arrival_date_day_of_month': [arrival_date_day],
        'stays_in_weekend_nights': [weekend_nights],
        'stays_in_week_nights': [week_nights],
        'lead_time': [lead_time]
    })
    
    # Apply the transformation
    try:
        # Transform the input data
        transformer = BookingDataTransformer()
        transformed_data = transformer.transform(input_data)
        
        # Drop the hotel column as it's constant for Resort Hotel
        if 'hotel' in transformed_data.columns:
            transformed_data = transformed_data.drop('hotel', axis=1)
        
        # Apply preprocessing
        processed_input = preprocessing_pipeline.transform(transformed_data)
        
        # Make prediction
        probabilities = model.predict_proba(processed_input)[0]
        cancellation_prob = probabilities[1] * 100
        
        # Show prediction
        st.subheader("Prediction Result")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Cancellation Probability", 
                f"{cancellation_prob:.2f}%",
                delta=f"{cancellation_prob - 50:.2f}%" if cancellation_prob != 50 else None,
                delta_color="inverse"
            )
            
            if cancellation_prob > 70:
                st.error("High risk of cancellation!")
            elif cancellation_prob > 40:
                st.warning("Moderate risk of cancellation")
            else:
                st.success("Low risk of cancellation")
                
        # Generate SHAP explanation
        st.subheader("Factors Influencing the Prediction")
        
        with st.spinner("Generating SHAP explanation..."):
            # Create a SHAP explainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(processed_input)
            
            # Create SHAP waterfall plot
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.plots._waterfall.waterfall_legacy(
                explainer.expected_value[1], 
                shap_values[1][0], 
                processed_input[0],
                feature_names=preprocessing_pipeline.get_feature_names_out(),
                show=False,
                max_display=15
            )
            st.pyplot(fig)
            
            st.info("""
            The waterfall plot shows how each feature contributes to pushing the model prediction 
            from the base value (average prediction) to the final prediction. 
            Red bars push the prediction higher (more likely to cancel), 
            while blue bars push the prediction lower (less likely to cancel).
            """)
    
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.write("Error details:", str(e))