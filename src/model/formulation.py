import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self):
        pass

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # New Feature Engineering Rules
        # 1. Compute other_screen allowing NaNs to propagate
        df['other_screen'] = df['daily_screen_time_hours'] - (df['social_media_hours'] + df['gaming_hours'] + df['work_study_hours'])

        # 2. Structural features
        df['unaccounted_hours'] = 24.0 - (df['daily_screen_time_hours'] + df['work_study_hours'] + df['sleep_hours'])
        df['cpr'] = df['app_opens_per_day'] / (df['notifications_per_day'] + 1.0)
        df['utl_ratio'] = df['other_screen'] / (df['daily_screen_time_hours'] + 1e-5)

        # work_shield_factor
        df['work_shield_factor'] = (df['work_study_hours'] / (df['daily_screen_time_hours'] + 1e-5)) * np.exp(- (df['social_media_hours'].fillna(0) + df['gaming_hours'].fillna(0)) / 2.0)

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # Domain Feature Engineering: Amount percentiles & cyclical transforms (from old implementation)
        for col in numeric_cols:
            if col not in ['id', 'target']:
                df[f'{col}_log'] = np.log1p(np.maximum(0, df[col].fillna(0)))
                # For percentiles, handle NaNs gracefully (rank natively handles NaNs by assigning NaN or omitting)
                df[f'{col}_pct'] = df[col].rank(pct=True)

        # 3. Memory downcasting: cast all engineered continuous floats strictly to np.float32
        engineered_cols = ['other_screen', 'unaccounted_hours', 'cpr', 'utl_ratio', 'work_shield_factor'] + \
                          [f'{col}_log' for col in numeric_cols if col not in ['id', 'target']] + \
                          [f'{col}_pct' for col in numeric_cols if col not in ['id', 'target']]

        for col in engineered_cols:
            if col in df.columns and df[col].dtype == np.float64:
                df[col] = df[col].astype(np.float32)

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit_transform(df)
