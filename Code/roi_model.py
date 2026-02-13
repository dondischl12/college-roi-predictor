import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

def load_processed_data():
    with open('data/processed/processed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

def calculate_roi(df):
    """Calculate actual ROI for each school"""
    
    # Calculate total education cost (4 years)
    df['total_education_cost'] = df['total_cost'] * 4
    
    # Calculate ROI percentage (simplified but realistic)
    WORKING_YEARS = 40
    df['lifetime_earnings'] = df['median_earnings_10yr'] * WORKING_YEARS
    df['roi_percentage'] = ((df['lifetime_earnings'] - df['total_education_cost']) / df['total_education_cost']) * 100
    
    # Calculate payback period
    avg_high_school_salary = 35000
    df['annual_earnings_premium'] = df['median_earnings_10yr'] - avg_high_school_salary
    df['payback_years'] = df['total_education_cost'] / df['annual_earnings_premium']
    
    return df

def build_roi_model(X_train, y_train):
    """Build Random Forest for ROI prediction"""
    print("Training Random Forest for ROI prediction...")
    
    rf_model = RandomForestRegressor(
        n_estimators=500,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='log2',
        bootstrap=True,
        max_samples=0.9,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    return rf_model

def evaluate_roi_model(model, X_train, y_train, X_test, y_test):
    """Evaluate ROI model performance"""
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print("\nROI Model Performance:")
    print("="*50)
    print(f"Training MAE:   {train_mae:.1f} percentage points")
    print(f"Test MAE:       {test_mae:.1f} percentage points")
    print(f"Training R²:    {train_r2:.3f}")
    print(f"Test R²:        {test_r2:.3f} ({test_r2*100:.1f}% accuracy)")
    
    overfitting_ratio = test_mae / train_mae
    if overfitting_ratio > 1.5:
        print(f"Overfitting detected: {overfitting_ratio:.2f}x")
    else:
        print(f"Good generalization! Ratio: {overfitting_ratio:.2f}")
    
    return {
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'y_test_pred': y_test_pred
    }

def analyze_roi_feature_importance(model, feature_names, top_n=10):
    """Analyze ROI feature importance"""
    
    importances = model.feature_importances_
    
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop {top_n} ROI Prediction Features:")
    print("="*50)
    for i, row in feature_importance_df.head(top_n).iterrows():
        print(f"{row['feature']:20s}: {row['importance']:.3f}")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    sns.barplot(data=feature_importance_df.head(top_n), x='importance', y='feature', 
                palette='viridis', hue='feature', legend=False)
    plt.title('Random Forest Feature Importance - College ROI Prediction', fontsize=16, pad=20)
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.tight_layout()
    plt.savefig('data/processed/roi_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return feature_importance_df

def create_roi_prediction_plots(y_test, y_pred):
    """Create ROI prediction analysis plots"""
    
    # 1. ROI Predictions vs Actual
    plt.figure(figsize=(12, 8))
    plt.scatter(y_test, y_pred, alpha=0.6, s=60, color='green', edgecolors='white', linewidth=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('Actual ROI (%)', fontsize=12)
    plt.ylabel('Predicted ROI (%)', fontsize=12)
    plt.title('College ROI: Predictions vs Reality', fontsize=16, pad=20)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('data/processed/roi_predictions_vs_actual.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. ROI Residuals Plot
    residuals = y_test - y_pred
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_pred, residuals, alpha=0.6, color='orange')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.xlabel('Predicted ROI (%)')
    plt.ylabel('Residuals (%)')
    plt.title('ROI Residuals vs Predicted Values')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(residuals, bins=30, color='lightgreen', alpha=0.7, edgecolor='black')
    plt.xlabel('ROI Prediction Error (%)')
    plt.ylabel('Frequency')
    plt.title('Distribution of ROI Prediction Errors')
    plt.axvline(x=0, color='black', linestyle='--')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/processed/roi_residual_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def show_roi_leaders(df):
    """Show top ROI schools and analysis"""
    
    print("\nTOP 15 SCHOOLS BY ROI:")
    print("="*80)
    roi_leaders = df.nlargest(15, 'roi_percentage')
    for i, (_, school) in enumerate(roi_leaders.iterrows(), 1):
        print(f"{i:2d}. {school['name'][:40]:<40} ({school['state']})")
        print(f"    Cost: ${school['total_education_cost']:>7,.0f} | Earnings: ${school['median_earnings_10yr']:>7,.0f}")
        print(f"    ROI: {school['roi_percentage']:>6.1f}% | Payback: {school['payback_years']:.1f} years")
        print()
    
    print("\nTOP 10 FASTEST PAYBACK (Best Value):")
    print("="*80)
    fast_payback = df[(df['payback_years'] > 0) & (df['payback_years'] < 20)].nsmallest(10, 'payback_years')
    for i, (_, school) in enumerate(fast_payback.iterrows(), 1):
        print(f"{i:2d}. {school['name'][:40]:<40} ({school['state']})")
        print(f"    Payback: {school['payback_years']:>4.1f} years | Cost: ${school['total_education_cost']:>7,.0f}")
        print(f"    ROI: {school['roi_percentage']:>6.1f}% | Earnings: ${school['median_earnings_10yr']:>7,.0f}")
        print()

def main():
    print("College ROI Predictor with Visualizations")
    print("="*60)
    
    # Load data
    data = load_processed_data()
    full_data = data['full_data']
    
    # Calculate ROI
    full_data = calculate_roi(full_data)
    
    # Show ROI leaders
    show_roi_leaders(full_data)
    
    # Prepare features for ROI prediction
    feature_cols = [
        'ownership', 'student_size', 'admission_rate', 'sat_average', 'act_midpoint',
        'tuition_in_state', 'tuition_out_state', 'total_cost', 'completion_rate_4yr',
        'state_encoded', 'tuition_ratio', 'selectivity_score', 'cost_per_completion'
    ]
    
    # Filter valid data
    valid_data = full_data[
        (full_data['roi_percentage'] > -100) & 
        (full_data['roi_percentage'] < 10000) & 
        (full_data['payback_years'] > 0) &
        (full_data['payback_years'] < 50)
    ].copy()
    
    print(f"Valid schools for ROI modeling: {len(valid_data)}")
    
    # Split data
    X = valid_data[feature_cols]
    y_roi = valid_data['roi_percentage']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_roi, test_size=0.2, random_state=42
    )
    
    # Build and evaluate model
    model = build_roi_model(X_train, y_train)
    results = evaluate_roi_model(model, X_train, y_train, X_test, y_test)
    
    # Feature importance analysis
    analyze_roi_feature_importance(model, feature_cols)
    
    # Create prediction plots
    create_roi_prediction_plots(y_test, results['y_test_pred'])
    
    # Save model
    roi_model_data = {
        'model': model,
        'feature_names': feature_cols,
        'model_performance': results
    }
    
    with open('data/processed/roi_model.pkl', 'wb') as f:
        pickle.dump(roi_model_data, f)
    
    print("\nROI model complete! Check PNG files in data/processed/")

if __name__ == "__main__":
    main()
