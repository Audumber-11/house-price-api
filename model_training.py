import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not installed. Using GradientBoosting instead.")


class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def prepare_data(self, df, target_col='MedHouseVal', test_size=0.2):
        feature_cols = [col for col in df.columns if col != target_col]
        self.feature_names = feature_cols
        
        X = df[feature_cols]
        y = df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_xgboost(self, X_train, y_train):
        if not HAS_XGB:
            return None
            
        print("\n[XGBoost Training]")
        model = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        param_grid = {
            'n_estimators': [100],
            'max_depth': [4, 6],
        }
        
        grid_search = GridSearchCV(
            model, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        self.models['xgboost'] = grid_search.best_estimator_
        print(f"Best XGBoost params: {grid_search.best_params_}")
        print(f"Best CV R² Score: {grid_search.best_score_:.4f}")
        
        return self.models['xgboost']
    
    def train_random_forest(self, X_train, y_train):
        print("\n[Random Forest Training]")
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        param_grid = {
            'n_estimators': [100],
            'max_depth': [6],
        }
        
        grid_search = GridSearchCV(
            model, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        self.models['random_forest'] = grid_search.best_estimator_
        print(f"Best RF params: {grid_search.best_params_}")
        print(f"Best CV R² Score: {grid_search.best_score_:.4f}")
        
        return self.models['random_forest']
    
    def train_gradient_boosting(self, X_train, y_train):
        print("\n[Gradient Boosting Training]")
        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        
        param_grid = {
            'n_estimators': [100],
            'max_depth': [4, 6],
        }
        
        grid_search = GridSearchCV(
            model, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        self.models['gradient_boosting'] = grid_search.best_estimator_
        print(f"Best GB params: {grid_search.best_params_}")
        print(f"Best CV R² Score: {grid_search.best_score_:.4f}")
        
        return self.models['gradient_boosting']
    
    def evaluate_all(self, X_test, y_test):
        results = {}
        
        print("\n" + "=" * 60)
        print("MODEL EVALUATION RESULTS")
        print("=" * 60)
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            cv_scores = cross_val_score(model, X_test, y_test, cv=3, scoring='r2')
            
            results[name] = {
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'CV_R2_Mean': cv_scores.mean(),
                'CV_R2_Std': cv_scores.std()
            }
            
            print(f"\n{name.upper().replace('_', ' ')}:")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE:  {mae:.4f}")
            print(f"  R²:   {r2:.4f}")
            print(f"  CV R² Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        best_model_name = max(results, key=lambda x: results[x]['R2'])
        self.best_model = self.models[best_model_name]
        
        print(f"\n*** Best Model: {best_model_name.upper()} ***")
        
        return results, best_model_name
    
    def get_feature_importance(self):
        if self.best_model is None:
            print("No model trained yet")
            return None
            
        importances = self.best_model.feature_importances_
        feature_imp = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feature_imp.to_string(index=False))
        
        return feature_imp
    
    def save_model(self, path='models'):
        import os
        os.makedirs(path, exist_ok=True)
        
        joblib.dump(self.scaler, f'{path}/scaler.pkl')
        joblib.dump(self.best_model, f'{path}/model.pkl')
        joblib.dump(self.feature_names, f'{path}/feature_names.pkl')
        
        print(f"\nModel saved to {path}/")
        print(f"  - scaler.pkl")
        print(f"  - model.pkl")
        print(f"  - feature_names.pkl")
    
    def load_model(self, path='models'):
        self.scaler = joblib.load(f'{path}/scaler.pkl')
        self.best_model = joblib.load(f'{path}/model.pkl')
        self.feature_names = joblib.load(f'{path}/feature_names.pkl')
        print(f"Model loaded from {path}/")
    
    def predict(self, features):
        if self.best_model is None:
            raise ValueError("No model loaded or trained")
            
        features_scaled = self.scaler.transform(features)
        return self.best_model.predict(features_scaled)


def main():
    from data_cleaning import main as load_processed_data
    
    print("=" * 60)
    print("HOUSE PRICE PREDICTION - MODEL TRAINING")
    print("=" * 60)
    
    df, _, _ = load_processed_data()
    print(f"\nLoaded data shape: {df.shape}")
    
    trainer = ModelTrainer()
    X_train, X_test, y_train, y_test = trainer.prepare_data(df)
    
    if HAS_XGB:
        trainer.train_xgboost(X_train, y_train)
    trainer.train_random_forest(X_train, y_train)
    trainer.train_gradient_boosting(X_train, y_train)
    
    results, best_model_name = trainer.evaluate_all(X_test, y_test)
    trainer.get_feature_importance()
    trainer.save_model()
    
    return trainer, results


if __name__ == "__main__":
    main()