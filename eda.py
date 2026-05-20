import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


class EDA:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def basic_stats(self):
        print("=" * 60)
        print("EXPLORATORY DATA ANALYSIS")
        print("=" * 60)
        print(f"\nDataset Shape: {self.df.shape}")
        print(f"\nColumn Types:\n{self.df.dtypes}")
        print(f"\nSummary Statistics:\n{self.df.describe()}")
        return self
    
    def correlation_analysis(self, target='MedHouseVal'):
        print(f"\nCorrelation with {target}:")
        corr = self.df.corr()[target].sort_values(ascending=False)
        print(corr)
        return self
    
    def create_visualizations(self, output_path='plots'):
        plt.style.use('seaborn-v0_8-whitegrid')
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        axes[0, 0].hist(self.df['MedHouseVal'], bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Median House Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of House Values')
        
        axes[0, 1].scatter(self.df['MedInc'], self.df['MedHouseVal'], alpha=0.3)
        axes[0, 1].set_xlabel('Median Income')
        axes[0, 1].set_ylabel('Median House Value')
        axes[0, 1].set_title('Income vs House Value')
        
        axes[0, 2].scatter(self.df['HouseAge'], self.df['MedHouseVal'], alpha=0.3)
        axes[0, 2].set_xlabel('House Age')
        axes[0, 2].set_ylabel('Median House Value')
        axes[0, 2].set_title('House Age vs Value')
        
        axes[1, 0].scatter(self.df['AveRooms'], self.df['MedHouseVal'], alpha=0.3)
        axes[1, 0].set_xlabel('Average Rooms')
        axes[1, 0].set_ylabel('Median House Value')
        axes[1, 0].set_title('Rooms vs House Value')
        
        axes[1, 1].scatter(self.df['Latitude'], self.df['Longitude'], 
                          c=self.df['MedHouseVal'], cmap='viridis', alpha=0.5)
        axes[1, 1].set_xlabel('Latitude')
        axes[1, 1].set_ylabel('Longitude')
        axes[1, 1].set_title('Geographic Distribution')
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        corr_matrix = self.df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   ax=axes[1, 2], fmt='.1f', annot_kws={'size': 6})
        axes[1, 2].set_title('Correlation Heatmap')
        
        plt.tight_layout()
        plt.savefig(f'{output_path}/eda_visualizations.png', dpi=150, bbox_inches='tight')
        print(f"Saved visualizations to {output_path}/eda_visualizations.png")
        plt.close()
        return self


def main():
    from data_cleaning import main as load_processed_data
    
    df, _, _ = load_processed_data()
    eda = EDA(df)
    eda.basic_stats()
    eda.correlation_analysis()
    eda.create_visualizations()


if __name__ == "__main__":
    main()