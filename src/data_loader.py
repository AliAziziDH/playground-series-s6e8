import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self):
        pass

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Domain Feature Engineering: Amount percentiles & cyclical transforms
        for col in numeric_cols:
            if col != 'id' and col != 'target':
                df[f'{col}_log'] = np.log1p(np.maximum(0, df[col]))
                df[f'{col}_pct'] = df[col].rank(pct=True)
                
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit_transform(df)
