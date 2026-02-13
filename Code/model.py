import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def load_processed_data(filepath='data/processed/processed_data.pkl'):
    """Load the processed data"""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    print("Loaded processed data:")
    print(f"Training set: {data['X_train'].shape}")
    print(f"Test set: {data['X_test'].shape}")
    print(f"Features: {len(data['feature_names'])}")
    
    return data

def build_random_forest(X_train, y_train):
    """Build Random Forest with overfitting fixes"""
    print("Training Random Forest with overfitting fixes...")
    
    # Better parameters to reduce overfitting
    rf_model = RandomForestRegressor(
        n_estimators=300,        # More trees for stability
        max_depth=15,            # Limit depth to prevent overfitting
        min_samples_split=10,    # Require more samples to split
        min_samples_leaf=5,      # Require more samples in each leaf
        max_features='sqrt',     # Use subset of features
        bootstrap=True,          # Use bootstrap sampling
        max_samples=0.8,         # Use 80% of samples per tree
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    return rf_model

def evaluate_model(model, X_train, y_train, X_test, y_test, feature_names):
    """Evaluate model performance"""
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print("\nImproved Model Performance:")
    print("="*50)
    print(f"Training MAE:   ${train_mae:,.0f}")
    print(f"Test MAE:       ${test_mae:,.0f}")
    print(f"Training RMSE:  ${train_rmse:,.0f}")
    print(f"Test RMSE:      ${test_rmse:,.0f}")
    print(f"Training R²:    {train_r2:.3f}")
    print(f"Test R²:        {test_r2:.3f}")
    
    # Check overfitting
    overfitting_ratio = test_mae / train_mae
    if overfitting_ratio > 1.5:
        print(f"\nStill some overfitting: {overfitting_ratio:.2f}x")
    else:
        print(f"\nGood generalization! Overfitting ratio: {overfitting_ratio:.2f}")
    
    return {
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'y_test_pred': y_test_pred,
        'overfitting_ratio': overfitting_ratio
    }

def analyze_feature_importance(model, feature_names, top_n=10):
    """Analyze and visualize feature importance"""
    
    importances = model.feature_importances_
    
    # Create DataFrame
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop {top_n} Most Important Features:")
    print("="*50)
    for i, row in feature_importance_df.head(top_n).iterrows():
        print(f"{row['feature']:20s}: {row['importance']:.3f}")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    sns.barplot(data=feature_importance_df.head(top_n), x='importance', y='feature', palette='viridis')
    plt.title('Random Forest Feature Importance - College ROI Predictors', fontsize=16, pad=20)
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.tight_layout()
    plt.savefig('data/processed/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return feature_importance_df

def create_prediction_plots(y_test, y_pred, full_data):
    """Create multiple prediction analysis plots"""
    
    # 1. Predictions vs Actual Scatter Plot
    plt.figure(figsize=(12, 8))
    plt.scatter(y_test, y_pred, alpha=0.6, s=60, color='steelblue', edgecolors='white', linewidth=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('Actual 10-Year Median Earnings ($)', fontsize=12)
    plt.ylabel('Predicted 10-Year Median Earnings ($)', fontsize=12)
    plt.title('College Earnings: Predictions vs Reality', fontsize=16, pad=20)
    plt.legend()
    
    # Format axes as currency
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('data/processed/predictions_vs_actual.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Residuals Plot
    residuals = y_test - y_pred
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_pred, residuals, alpha=0.6, color='coral')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.xlabel('Predicted Earnings ($)')
    plt.ylabel('Residuals ($)')
    plt.title('Residuals vs Predicted Values')
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(residuals, bins=30, color='lightcoral', alpha=0.7, edgecolor='black')
    plt.xlabel('Prediction Error ($)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    plt.axvline(x=0, color='black', linestyle='--')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/processed/residual_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def show_top_schools(X_test, y_test, y_pred, full_data):
    """Show top performing schools and biggest prediction errors"""
    
    # Get test school info
    test_indices = X_test.index
    test_schools = full_data.loc[test_indices].copy()
    test_schools['predicted_earnings'] = y_pred
    test_schools['actual_earnings'] = y_test.values
    test_schools['prediction_error'] = test_schools['actual_earnings'] - test_schools['predicted_earnings']
    
    print("\nTOP 10 HIGHEST EARNING SCHOOLS:")
    print("="*70)
    top_earners = test_schools.nlargest(10, 'actual_earnings')
    for i, (_, school) in enumerate(top_earners.iterrows(), 1):
        error = school['prediction_error']
        error_sign = "+" if error >= 0 else ""
        print(f"{i:2d}. {school['name'][:35]:<35} ({school['state']})")
        print(f"    Actual: ${school['actual_earnings']:>7,.0f} | Predicted: ${school['predicted_earnings']:>7,.0f} | Error: {error_sign}${error:>6,.0f}")
        print()
    
    print("\nBIGGEST PREDICTION ERRORS:")
    print("="*70)
    print("OVERESTIMATES (Model too optimistic):")
    overestimates = test_schools.nsmallest(5, 'prediction_error')
    for i, (_, school) in enumerate(overestimates.iterrows(), 1):
        print(f"{i}. {school['name'][:35]:<35} ({school['state']})")
        print(f"   Predicted: ${school['predicted_earnings']:>7,.0f} | Actual: ${school['actual_earnings']:>7,.0f} | Off by: ${abs(school['prediction_error']):>6,.0f}")
        print()
    
    print("UNDERESTIMATES (Model too pessimistic):")
    underestimates = test_schools.nlargest(5, 'prediction_error')
    for i, (_, school) in enumerate(underestimates.iterrows(), 1):
        print(f"{i}. {school['name'][:35]:<35} ({school['state']})")
        print(f"   Predicted: ${school['predicted_earnings']:>7,.0f} | Actual: ${school['actual_earnings']:>7,.0f} | Off by: ${school['prediction_error']:>6,.0f}")
        print()

def save_model(model, feature_names, filepath='data/processed/trained_model.pkl'):
    """Save the trained model"""
    model_data = {
        'model': model,
        'feature_names': feature_names,
        'model_type': 'RandomForestRegressor'
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved to {filepath}")

def main():
    print("Building Improved Random Forest College ROI Predictor")
    print("="*60)
    
    # Load processed data
    data = load_processed_data()
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']
    feature_names = data['feature_names']
    full_data = data['full_data']
    
    # Build improved model
    model = build_random_forest(X_train, y_train)
    
    # Evaluate model
    results = evaluate_model(model, X_train, y_train, X_test, y_test, feature_names)
    
    # Analyze feature importance
    feature_importance_df = analyze_feature_importance(model, feature_names)
    
    # Create prediction plots
    create_prediction_plots(y_test, results['y_test_pred'], full_data)
    
    # Show top schools and errors
    show_top_schools(X_test, y_test, results['y_test_pred'], full_data)
    
    # Save model
    save_model(model, feature_names)
    
    print("\nImproved model training complete!")
    print("Check the PNG files in data/processed/ for visualizations")

if __name__ == "__main__":
    main()
