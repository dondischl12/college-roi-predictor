import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_clean_data(filepath='data/raw/college_data_raw.csv'):
    """Load and preprocess the college data"""
    df = pd.read_csv(filepath)
    
    print(f"Loaded {len(df)} schools")
    
    # Filter for schools with earnings data (our target variable)
    df_clean = df[df['median_earnings_10yr'].notna()].copy()
    print(f"Schools with earnings data: {len(df_clean)}")
    
    # Handle missing values strategically
    df_clean['admission_rate'] = df_clean['admission_rate'].fillna(df_clean['admission_rate'].median())
    
    # Fill test scores by ownership type (public vs private)
    for ownership_type in df_clean['ownership'].unique():
        if pd.isna(ownership_type):
            continue
        mask = df_clean['ownership'] == ownership_type
        df_clean.loc[mask, 'sat_average'] = df_clean.loc[mask, 'sat_average'].fillna(
            df_clean.loc[mask, 'sat_average'].median()
        )
        df_clean.loc[mask, 'act_midpoint'] = df_clean.loc[mask, 'act_midpoint'].fillna(
            df_clean.loc[mask, 'act_midpoint'].median()
        )
    
    # Fill remaining missing values with overall median
    numeric_cols = ['student_size', 'tuition_in_state', 'tuition_out_state', 
                   'total_cost', 'completion_rate_4yr', 'sat_average', 'act_midpoint']
    
    for col in numeric_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # Create engineered features
    df_clean['tuition_ratio'] = df_clean['tuition_out_state'] / df_clean['tuition_in_state']
    df_clean['selectivity_score'] = 1 - df_clean['admission_rate']  # Higher = more selective
    df_clean['cost_per_completion'] = df_clean['total_cost'] / (df_clean['completion_rate_4yr'] + 0.01)  # Avoid division by zero
    
    # Encode categorical variables
    le_state = LabelEncoder()
    df_clean['state_encoded'] = le_state.fit_transform(df_clean['state'])
    
    print(f"Final clean dataset shape: {df_clean.shape}")
    print(f"Target variable (earnings) range: ${df_clean['median_earnings_10yr'].min():,.0f} - ${df_clean['median_earnings_10yr'].max():,.0f}")
    
    return df_clean, le_state

def prepare_features_target(df):
    """Prepare feature matrix and target variable for modeling"""
    
    feature_cols = [
        'ownership',           # Public vs Private (1=Public, 2=Private nonprofit, 3=Private for-profit)
        'student_size',        # School size
        'admission_rate',      # Selectivity
        'sat_average',         # Academic quality
        'act_midpoint',        # Academic quality
        'tuition_in_state',    # Cost
        'tuition_out_state',   # Cost
        'total_cost',          # Total cost of attendance
        'completion_rate_4yr', # Graduation success rate
        'state_encoded',       # Location (encoded)
        'tuition_ratio',       # Out-of-state premium
        'selectivity_score',   # Selectivity (1 - admission_rate)
        'cost_per_completion'  # Cost efficiency
    ]
    
    X = df[feature_cols].copy()
    y = df['median_earnings_10yr'].copy()
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Features: {list(X.columns)}")
    
    return X, y

def main():
    # Load and clean data
    df_clean, le_state = load_and_clean_data()
    
    # Show some summary stats
    print("\nDataset Summary:")
    print(f"Ownership breakdown:")
    ownership_map = {1: 'Public', 2: 'Private nonprofit', 3: 'Private for-profit'}
    for code, count in df_clean['ownership'].value_counts().items():
        print(f"  {ownership_map.get(code, f'Code {code}')}: {count} schools")
    
    print(f"\nTop 10 states by school count:")
    print(df_clean['state'].value_counts().head(10))
    
    # Prepare features and target
    X, y = prepare_features_target(df_clean)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=df_clean['ownership']
    )
    
    print(f"\nData split:")
    print(f"Training set: {X_train.shape[0]} schools")
    print(f"Test set: {X_test.shape[0]} schools")
    
    # Save processed data
    processed_data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': list(X.columns),
        'label_encoders': {'state': le_state},
        'full_data': df_clean
    }
    
    import pickle
    with open('data/processed/processed_data.pkl', 'wb') as f:
        pickle.dump(processed_data, f)
    
    print("\nProcessed data saved to data/processed/processed_data.pkl")
    print("Ready for Random Forest modeling!")

if __name__ == "__main__":
    main()
