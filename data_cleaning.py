import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing


class DataCleaning:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def handle_missing_values(self):
        missing = self.df.isnull().sum()
        print("Missing values per column:")
        print(missing[missing > 0] if missing.sum() > 0 else "No missing values found")
        
        for col in self.df.columns:
            if self.df[col].isnull().sum() > 0:
                if self.df[col].dtype in ['float64', 'int64']:
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                else:
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        return self
    
    def remove_outliers(self, columns=None, method='iqr', threshold=1.5):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if 'MedHouseVal' in columns:
            columns.remove('MedHouseVal')
            
        removed = 0
        for col in columns:
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
                mask = (self.df[col] >= lower) & (self.df[col] <= upper)
                removed += (~mask).sum()
                self.df = self.df[mask]
                
        print(f"Removed {removed} outliers using IQR method")
        return self
    
    def create_features(self):
        self.df['RoomsPerHousehold'] = self.df['AveRooms'] / self.df['AveOccup']
        self.df['BedroomsRatio'] = self.df['AveBedrms'] / self.df['AveRooms']
        self.df['PopulationPerOccup'] = self.df['Population'] / self.df['AveOccup']
        
        self.df['MedInc_squared'] = self.df['MedInc'] ** 2
        self.df['HouseAge_squared'] = self.df['HouseAge'] ** 2
        
        self.df['MedInc_HouseAge'] = self.df['MedInc'] * self.df['HouseAge']
        self.df['MedInc_AveRooms'] = self.df['MedInc'] * self.df['AveRooms']
        
        self.df['Location_Score'] = (
            (40 - self.df['Latitude']).abs() + 
            (120 - self.df['Longitude']).abs()
        )
        
        return self
    
    def get_processed_data(self) -> pd.DataFrame:
        return self.df


def load_data():
    housing = fetch_california_housing(as_frame=True)
    return housing.frame, housing.feature_names, housing.target_names


def main():
    df, feature_names, target_names = load_data()
    print(f"Original data shape: {df.shape}")
    
    cleaner = DataCleaning(df)
    cleaner.handle_missing_values()
    cleaner.remove_outliers()
    cleaner.create_features()
    
    processed_df = cleaner.get_processed_data()
    print(f"Processed data shape: {processed_df.shape}")
    print(f"\nNew features created:")
    print([col for col in processed_df.columns if col not in df.columns])
    
    return processed_df, feature_names, target_names


if __name__ == "__main__":
    main()