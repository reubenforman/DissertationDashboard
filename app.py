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
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS to improve appearance
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
    }
    .stMetric {
        background-color: #f7f7f7;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    .css-1v0mbdj.e115fcil1 {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

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
        df['portugal'] = (df['country'] == 'PRT').astype(int)
        df['european'] = df['country'].isin(european_countries).astype(int)
        df['rest_of_the_world'] = ((~df['country'].isin(european_countries)) & (df['country'] != 'PRT')).astype(int)
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

# Create tabs for better organization
tab1, tab2 = st.tabs(["📝 Booking Information", "📊 Advanced Options"])

with tab1:
    # Create a more visually appealing input form
    st.subheader("Enter Booking Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Guest Information")
        adults = st.number_input("Number of Adults", min_value=1, max_value=10, value=2)
        children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
        babies = st.number_input("Number of Babies", min_value=0, max_value=10, value=0)
        is_repeated_guest = st.selectbox("Is Repeated Guest", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        country = st.selectbox("Country", options=["PRT", "GBR", "FRA", "ESP", "DEU", "ITA", "USA", "Other"])
        if country == "Other":
            country = st.text_input("Specify Country Code (3 letters)")
        
        st.markdown("### Booking Details")
        adr = st.number_input("Average Daily Rate (EUR)", min_value=0, max_value=1000, value=100, step=1)
        lead_time = st.number_input("Lead Time (days)", min_value=0, max_value=365, value=30)
    
    with col2:
        st.markdown("### Stay Information")
        weekend_nights = st.number_input("Stays in Weekend Nights", min_value=0, max_value=10, value=1)
        week_nights = st.number_input("Stays in Week Nights", min_value=0, max_value=30, value=3)
        meal = st.selectbox("Meal Plan", options=["BB", "FB", "HB", "SC", "Undefined"], 
                            format_func=lambda x: {
                                "BB": "Bed & Breakfast",
                                "FB": "Full Board",
                                "HB": "Half Board",
                                "SC": "Self Catering",
                                "Undefined": "Not Specified"
                            }.get(x, x))
        
        st.markdown("### Room Information")
        reserved_room_type = st.selectbox("Reserved Room Type", options=["A", "B", "C", "D", "E", "F", "G", "H"])
        assigned_room_type = st.selectbox("Assigned Room Type", options=["A", "B", "C", "D", "E", "F", "G", "H"])
        
        st.markdown("### Arrival Date")
        arrival_date_year = st.selectbox("Arrival Year", options=[2015, 2016, 2017])
        arrival_date_month = st.selectbox("Arrival Month", options=[
            "January", "February", "March", "April", "May", "June", 
            "July", "August", "September", "October", "November", "December"
        ])
        arrival_date_day = st.number_input("Arrival Day", min_value=1, max_value=31, value=15)

with tab2:
    st.subheader("Advanced Booking Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Booking Details")
        market_segment = st.selectbox("Market Segment", options=["Direct", "Corporate", "Online TA", "Offline TA/TO", "Complementary", "Groups", "Aviation"])
        distribution_channel = st.selectbox("Distribution Channel", options=["Direct", "Corporate", "TA/TO", "GDS", "Undefined"])
        deposit_type = st.selectbox("Deposit Type", options=["No Deposit", "Non Refund", "Refundable"])
        customer_type = st.selectbox("Customer Type", options=["Transient", "Transient-Party", "Contract", "Group"])
        agent = st.number_input("Agent ID", min_value=0, max_value=535, value=0)
        company = st.number_input("Company ID", min_value=0, max_value=500, value=0)
    
    with col2:
        st.markdown("### Booking History")
        previous_cancellations = st.number_input("Previous Cancellations", min_value=0, max_value=50, value=0)
        previous_bookings_not_canceled = st.number_input("Previous Bookings Not Canceled", min_value=0, max_value=50, value=0)
        booking_changes = st.number_input("Booking Changes", min_value=0, max_value=20, value=0)
        days_in_waiting_list = st.number_input("Days in Waiting List", min_value=0, max_value=365, value=0)
        required_car_parking_spaces = st.number_input("Required Car Parking Spaces", min_value=0, max_value=8, value=0)
        total_of_special_requests = st.number_input("Total of Special Requests", min_value=0, max_value=5, value=0)
        arrival_date_week_number = st.number_input("Arrival Date Week Number", min_value=1, max_value=53, value=25)

# Add a divider for visual separation
st.divider()

# Create a more prominent predict button
predict_button = st.button("Predict Cancellation Probability", type="primary", use_container_width=True)

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
        # Create a spinner for better UX during processing
        with st.spinner("Processing your booking information..."):
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
        
        # Show prediction in a more visually appealing way
        st.subheader("Prediction Result")
        
        # Create columns for prediction result and details
        col1, col2 = st.columns([1, 2])
        
        with col1:       
            st.metric(
                "Cancellation Probability", 
                f"{cancellation_prob:.1f}%",
                delta=f"{cancellation_prob - 50:.1f}%" if cancellation_prob != 50 else None,
                delta_color="inverse"
            )
            
            if cancellation_prob > 70:
                st.error("❗ High risk of cancellation!")
            elif cancellation_prob > 40:
                st.warning("⚠️ Moderate risk of cancellation")
            else:
                st.success("✅ Low risk of cancellation")
                
        # Generate SHAP explanation
        st.subheader("Factors Influencing the Prediction")

        with st.spinner("Generating explanation..."):
            # Define feature name mapping for better readability
            feature_name_map = {
                'adults': 'Adults',
                'adr': 'Average Daily Rate',
                'lead_time': 'Lead Time',
                'total_stay_nights': 'Total Nights Stayed',
                'market_segment': 'Market Segment',
                'distribution_channel': 'Distribution Channel',
                'portugal': 'Country: Portugal',
                'european': 'Country: European',
                'rest_of_the_world': 'Country: Rest of World',
                'deposit_type': 'Deposit Type',
                'customer_type': 'Customer Type',
                'is_repeated_guest': 'Is Repeated Guest',
                'required_car_spaces_indicator': 'Car Parking Required',
                'room_status': 'Reservation Room Match',
                'season': 'Season',
                'is_family': 'Family',
                'cancellation_risk': 'Previous Cancellation History',
                'special_request_indicator': 'Special Request Made',
                'waiting_list_indicator': 'On Waiting List',
                'agent': 'Agent',
                'arrival_day_name': 'Day of Arrival',
                'arrival_date_year': 'Arrival Year',
                'booking_changes_indicator': 'Booking Changes Made',
                'meal': 'Meal Plan',
                'previous_bookings_not_canceled': 'Previous Bookings Not Canceled'
            }
            
            # Create a SHAP explainer
            explainer = shap.TreeExplainer(model)
            
            # Ensure processed_input is a 2D array
            if len(processed_input.shape) == 1:
                processed_input_2d = processed_input.reshape(1, -1)
            else:
                processed_input_2d = processed_input
                
            # Get SHAP values
            shap_values = explainer.shap_values(processed_input_2d)
            
            # Get expected value - handle both scalar and array cases
            if hasattr(explainer.expected_value, "__len__"):
                expected_value = explainer.expected_value[1]
            else:
                expected_value = explainer.expected_value
                
            # Get appropriate SHAP values
            if isinstance(shap_values, list):
                shap_values_to_plot = shap_values[1][0]
            else:
                shap_values_to_plot = shap_values[0]
            
            # Get encoded feature names
            encoded_feature_names = preprocessing_pipeline.get_feature_names_out()
            
            # Improved aggregation of SHAP values for binary/categorical features
            feature_groups = {}
            
            for i, feature in enumerate(encoded_feature_names):
                # Skip features with negligible impact
                if abs(shap_values_to_plot[i]) < 1e-10:
                    continue
                    
                # Extract base feature name by removing prefixes/suffixes
                base_feature = feature
                feature_value = None
                
                if feature.startswith('cat__'):
                    parts = feature.split('__', 1)[1].rsplit('_', 1)
                    if len(parts) > 1:
                        base_feature = parts[0]
                        feature_value = parts[1]
                elif feature.startswith('num__'):
                    base_feature = feature.split('__', 1)[1]
                
                # Get readable name
                readable_base = feature_name_map.get(base_feature, base_feature)
                
                # Handle binary features
                if base_feature in (
                    'cancellation_risk', 'is_repeated_guest', 'required_car_spaces_indicator', 
                    'room_status', 'is_family', 'special_request_indicator', 
                    'waiting_list_indicator', 'booking_changes_indicator'
                ):
                    actual_value = None
                    # Determine the actual value from transformed data if available
                    if base_feature in transformed_data.columns:
                        actual_value = bool(transformed_data[base_feature].iloc[0])
                    elif feature_value:
                        if feature_value in ['1', 'True']:
                            actual_value = True
                        elif feature_value in ['0', 'False']:
                            actual_value = False
                    
                    if actual_value is not None:
                        display_name = f"{readable_base}: {'Yes' if actual_value else 'No'}"
                        # If this feature's one-hot matches the actual value
                        if (feature_value in ['1','True']) == actual_value:
                            if display_name not in feature_groups:
                                feature_groups[display_name] = shap_values_to_plot[i]
                            else:
                                feature_groups[display_name] += shap_values_to_plot[i]
                
                # Handle country features
                elif base_feature in ('portugal', 'european', 'rest_of_the_world'):
                    value = None
                    if base_feature in transformed_data.columns:
                        value = bool(transformed_data[base_feature].iloc[0])
                    elif feature_value:
                        value = (feature_value in ['1','True'])
                        
                    if value:
                        if readable_base not in feature_groups:
                            feature_groups[readable_base] = shap_values_to_plot[i]
                        else:
                            feature_groups[readable_base] += shap_values_to_plot[i]
                            
                # Handle categorical features
                elif feature_value:
                    display_name = f"{readable_base}: {feature_value}"
                    if display_name not in feature_groups:
                        feature_groups[display_name] = shap_values_to_plot[i]
                    else:
                        feature_groups[display_name] += shap_values_to_plot[i]
                
                # Handle numeric features (e.g., Lead Time)
                else:
                    if base_feature in transformed_data.columns:
                        numeric_val = transformed_data[base_feature].iloc[0]
                        # Optionally round numeric_val
                        display_name = f"{readable_base}: {numeric_val}"
                    else:
                        display_name = readable_base
                    
                    if display_name not in feature_groups:
                        feature_groups[display_name] = shap_values_to_plot[i]
                    else:
                        feature_groups[display_name] += shap_values_to_plot[i]
 
            # Get top features by absolute value
            sorted_features = sorted(
                feature_groups.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            )[:10]  # Top 10 features
            
            # Create the waterfall plot
            # fig, ax = plt.subplots(figsize=(6, 4))  # Reduced figure size
            plt.style.use('default')
            
            # Prepare features and values for the plot
            features = [x[0] for x in sorted_features]
            values = [x[1] for x in sorted_features]
            
            # SHAP waterfall plot
            fig = shap.plots._waterfall.waterfall_legacy(
                expected_value, 
                np.array(values), 
                feature_names=features,
                max_display=10,
                show=False
            )
            # Explicitly set the figure size
            fig.set_size_inches(8, 6)

            # Turn off the grid for all axes in the figure
            for ax in fig.axes:
                ax.grid(False) 
            # Improve plot appearance
            plt.title('Top Factors Affecting Cancellation Probability', fontsize=8)
            plt.tight_layout()
            
            st.pyplot(fig)
            
            # Add informative explanation box
            with st.expander("How to Interpret This Chart", expanded=True):
                st.markdown("""
                ### Understanding the Prediction Factors
                
                - **Red bars** indicate factors that **increase** the likelihood of cancellation
                - **Blue bars** indicate factors that **decrease** the likelihood of cancellation
                - The **size of each bar** shows how much influence that factor has on the prediction
                - The **starting value** (E[f(x)]) is the average prediction for all bookings
                - The **final value** is the predicted cancellation probability for this specific booking
                """)
            
            # Display the most important factors as a table
            st.subheader("Key Factors Summary")
            factor_impact = []
            for feature, value in sorted_features[:5]:  # Top 5 in the table
                impact = "Increases" if value > 0 else "Decreases"
                factor_impact.append({
                    "Factor": feature,
                    "Impact": impact,
                    "Strength": abs(value)
                })
            
            factor_df = pd.DataFrame(factor_impact)
            factor_df["Strength"] = factor_df["Strength"].round(3)
            st.table(factor_df)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.write("Error details:", str(e))
