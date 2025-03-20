import pandas as pd
import numpy as np
import joblib
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
np.random.seed(0)

# Ensure output directories exist
os.makedirs('dashboard/models', exist_ok=True)

# Define the transformer class
class BookingDataTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        # No hyperparameters needed here (but you could add some if desired)
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

# Main training function
def train_and_save_model():
    print("Loading data...")
    data = pd.read_excel('data/hotel.xlsx')
    
    print("Transforming data...")
    data_clean = BookingDataTransformer().fit_transform(data)
    H1 = data_clean[data_clean['hotel'] == 'Resort Hotel'].drop('hotel', axis=1)
    
    # Define features
    numerical = ['adults', 'adr', 'lead_time', 'total_stay_nights']
    categorical = [
        'market_segment', 'distribution_channel', 'country_of_origin',
        'deposit_type', 'customer_type', 'previous_bookings_not_canceled',
        'is_repeated_guest', 'required_car_spaces_indicator',
        'room_status', 'season', 'is_family', 'cancellation_risk',
        'special_request_indicator', 'waiting_list_indicator', 'agent',
        'arrival_day_name', 'arrival_date_year', 'booking_changes_indicator', 'meal'
    ]

    # Create preprocessing pipeline
    print("Creating preprocessing pipeline...")
    preprocessing_pipeline = Pipeline([
        ('preprocessor', ColumnTransformer([
            ('num', StandardScaler(), numerical),
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical)
        ]))
    ])

    # Split data
    print("Splitting data...")
    X = H1.drop('is_canceled', axis=1)
    y = H1['is_canceled']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Preprocess and balance data
    print("Preprocessing and applying SMOTE...")
    X_train_processed = preprocessing_pipeline.fit_transform(X_train)
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_processed, y_train)

    # Train final model
    print("Training the model...")
    best_params = {
        'max_depth': 14, 'num_leaves': 233, 'min_data_in_leaf': 26,
        'learning_rate': 0.02, 'n_estimators': 1424, 'subsample': 0.71,
        'colsample_bytree': 0.89, 'reg_alpha': 0.00003, 'reg_lambda': 0.0023
    }
    
    model = LGBMClassifier(**best_params, verbose=-1)
    model.fit(X_train_smote, y_train_smote)

    # Save artifacts
    print("Saving model and preprocessing pipeline...")
    joblib.dump(model, 'dashboard/models/best_model.pkl')
    joblib.dump(preprocessing_pipeline, 'dashboard/models/preprocessing_pipeline.pkl')
    print("Model trained and saved successfully!")

    # Also save a sample data record for testing
    X_sample = X.iloc[0:1].copy()
    joblib.dump(X_sample, 'dashboard/models/sample_input.pkl')
    print("Sample input saved for testing!")

if __name__ == "__main__":
    train_and_save_model()