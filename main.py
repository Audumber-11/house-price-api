"""
House Price Prediction Model
============================
A machine learning model to predict house prices using Linear Regression.
Dataset: California Housing Prices (sklearn.datasets)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


class HousePricePredictor:
    """A class to build and evaluate a house price prediction model."""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.target_name = None
        
    def load_data(self):
        """Load the California housing dataset."""
        print("=" * 60)
        print("HOUSE PRICE PREDICTION MODEL")
        print("=" * 60)
        print("\n[1] Loading California Housing Dataset...")
        
        housing = fetch_california_housing(as_frame=True)
        df = housing.frame
        
        print(f"    Dataset shape: {df.shape}")
        print(f"    Features: {housing.feature_names}")
        print(f"    Target: {housing.target_names}")
        
        self.feature_names = housing.feature_names
        self.target_name = housing.target_names[0]
        
        return df, housing
    
    def explore_data(self, df):
        """Explore the dataset."""
        print("\n[2] Data Exploration...")
        
        print("\n    First 5 rows:")
        print(df.head().to_string())
        
        print("\n    Dataset Statistics:")
        print(df.describe().to_string())
        
        print("\n    Missing Values:")
        print(df.isnull().sum().to_string())
        
        print("\n    Correlation with target (MedHouseVal):")
        correlations = df.corr()[self.target_name].sort_values(ascending=False)
        print(correlations.to_string())
        
        return df
    
    def preprocess_data(self, df):
        """Preprocess the data for training."""
        print("\n[3] Preprocessing Data...")
        
        X = df[self.feature_names]
        y = df[self.target_name]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"    Training set size: {len(self.X_train)}")
        print(f"    Test set size: {len(self.X_test)}")
        
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("    Data scaled using StandardScaler")
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def train_model(self):
        """Train the linear regression model."""
        print("\n[4] Training Linear Regression Model...")
        
        self.model = LinearRegression()
        self.model.fit(self.X_train_scaled, self.y_train)
        
        print("    Model trained successfully!")
        
        print("\n    Model Coefficients:")
        for feature, coef in zip(self.feature_names, self.model.coef_):
            print(f"      {feature}: {coef:.4f}")
        print(f"      Intercept: {self.model.intercept_:.4f}")
        
        return self.model
    
    def evaluate_model(self):
        """Evaluate the model performance."""
        print("\n[5] Model Evaluation...")
        
        y_pred_train = self.model.predict(self.X_train_scaled)
        y_pred_test = self.model.predict(self.X_test_scaled)
        
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
        
        train_mae = mean_absolute_error(self.y_train, y_pred_train)
        test_mae = mean_absolute_error(self.y_test, y_pred_test)
        
        train_r2 = r2_score(self.y_train, y_pred_train)
        test_r2 = r2_score(self.y_test, y_pred_test)
        
        print(f"\n    {'Metric':<25} {'Train':<15} {'Test':<15}")
        print(f"    {'-'*55}")
        print(f"    {'RMSE':<25} {train_rmse:<15.4f} {test_rmse:<15.4f}")
        print(f"    {'MAE':<25} {train_mae:<15.4f} {test_mae:<15.4f}")
        print(f"    {'R² Score':<25} {train_r2:<15.4f} {test_r2:<15.4f}")
        
        print("\n    Interpretation:")
        if test_r2 >= 0.8:
            print("      ✓ Excellent model! R² > 0.8")
        elif test_r2 >= 0.6:
            print("      ✓ Good model! R² > 0.6")
        elif test_r2 >= 0.4:
            print("      △ Moderate model. R² > 0.4")
        else:
            print("      ✗ Poor model. Needs improvement")
        
        print(f"\n    The model explains {test_r2*100:.1f}% of the variance in house prices.")
        print(f"    Average prediction error: ${test_mae * 100000:.0f}")
        
        return {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'predictions': y_pred_test
        }
    
    def predict_sample(self, sample_data=None):
        """Make predictions on sample data."""
        print("\n[6] Sample Predictions...")
        
        if sample_data is None:
            sample_data = {
                'MedInc': 5.0,
                'HouseAge': 20.0,
                'AveRooms': 6.0,
                'AveBedrms': 1.0,
                'Population': 1000.0,
                'AveOccup': 3.0,
                'Latitude': 34.0,
                'Longitude': -118.0
            }
        
        sample_df = pd.DataFrame([sample_data])
        sample_scaled = self.scaler.transform(sample_df)
        
        prediction = self.model.predict(sample_scaled)[0]
        
        print("    Sample House Features:")
        for key, value in sample_data.items():
            print(f"      {key}: {value}")
        print(f"\n    Predicted Price: ${prediction * 100000:.2f}")
        
        return prediction
    
    def visualize_results(self, y_test, y_pred):
        """Visualize the model results."""
        print("\n[7] Generating Visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        axes[0, 0].scatter(y_test, y_pred, alpha=0.5, edgecolors='k', linewidths=0.5)
        axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Prices (in $100,000s)')
        axes[0, 0].set_ylabel('Predicted Prices (in $100,000s)')
        axes[0, 0].set_title('Actual vs Predicted Prices')
        axes[0, 0].grid(True, alpha=0.3)
        
        residuals = y_test - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.5, edgecolors='k', linewidths=0.5)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Prices')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        feature_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Coefficient': np.abs(self.model.coef_)
        }).sort_values('Coefficient', ascending=True)
        
        axes[1, 0].barh(feature_importance['Feature'], feature_importance['Coefficient'], color='steelblue')
        axes[1, 0].set_xlabel('Absolute Coefficient')
        axes[1, 0].set_title('Feature Importance')
        axes[1, 0].grid(True, alpha=0.3, axis='x')
        
        axes[1, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel('Residuals')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of Residuals')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('house_price_analysis.png', dpi=150, bbox_inches='tight')
        print("    Visualization saved to: house_price_analysis.png")
        plt.close()
    
    def run(self):
        """Run the complete pipeline."""
        df, housing = self.load_data()
        self.explore_data(df)
        self.preprocess_data(df)
        self.train_model()
        metrics = self.evaluate_model()
        self.predict_sample()
        self.visualize_results(self.y_test, metrics['predictions'])
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE!")
        print("=" * 60)


if __name__ == "__main__":
    predictor = HousePricePredictor()
    predictor.run()