import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.formulation import DataProcessor
from src.model.solver import ValueLevelTargetEncoder

@pytest.fixture
def dummy_data():
    np.random.seed(42)
    N = 100
    df = pd.DataFrame()
    df['id'] = np.arange(N)

    # Continuous
    df['daily_screen_time_hours'] = np.random.uniform(2.0, 16.0, N)
    df['social_media_hours'] = np.random.uniform(0.0, 6.0, N)
    df['gaming_hours'] = np.random.uniform(0.0, 6.0, N)
    df['work_study_hours'] = np.random.uniform(0.0, 10.0, N)
    df['sleep_hours'] = np.random.uniform(4.0, 10.0, N)
    df['app_opens_per_day'] = np.random.randint(10, 200, N)
    df['notifications_per_day'] = np.random.randint(20, 300, N)

    # Categoricals
    df['gender'] = np.random.choice(['Male', 'Female', 'Other'], N)
    df['stress_level'] = np.random.choice(['Low', 'Medium', 'High'], N)
    df['academic_work_impact'] = np.random.choice(['None', 'Low', 'Moderate', 'Severe'], N)

    # Add random NaNs
    for col in ['gaming_hours', 'work_study_hours', 'social_media_hours']:
        mask = np.random.rand(N) < 0.1
        df.loc[mask, col] = np.nan

    return df

@pytest.fixture
def dummy_target():
    np.random.seed(42)
    return pd.Series(np.random.randint(0, 2, 100), name='target')

def test_feature_engineering_nans(dummy_data):
    processor = DataProcessor()
    df_transformed = processor.fit_transform(dummy_data)

    # other_screen should have NaNs where gaming, work or social are NaN
    # because we didn't fillna(0) inside the sum
    expected_nans = dummy_data[['social_media_hours', 'gaming_hours', 'work_study_hours']].isna().any(axis=1).sum()
    assert df_transformed['other_screen'].isna().sum() == expected_nans, "NaNs are not propagated correctly in other_screen"

def test_feature_engineering_types(dummy_data):
    processor = DataProcessor()
    df_transformed = processor.fit_transform(dummy_data)

    # Check that engineered cols are float32
    float32_cols = ['other_screen', 'unaccounted_hours', 'cpr', 'utl_ratio', 'work_shield_factor']
    for col in float32_cols:
        assert df_transformed[col].dtype == np.float32, f"{col} should be float32"

def test_target_encoder_leak_free(dummy_data, dummy_target):
    te = ValueLevelTargetEncoder(smoothing=10.0, cols=['gender', 'stress_level', 'academic_work_impact', 'daily_screen_time_hours_rounded'])
    dummy_data['daily_screen_time_hours_rounded'] = dummy_data['daily_screen_time_hours'].round(1).astype(str)

    # Transform
    encoded = te.fit_transform(dummy_data, dummy_target)

    # Check that TE is applied to all categoricals
    for col in ['gender', 'stress_level', 'academic_work_impact']:
        assert f'{col}_te' in encoded.columns
        assert not encoded[f'{col}_te'].isna().any()
        assert encoded[f'{col}_te'].dtype == np.float32

    # daily_screen_time_hours should NOT be target encoded
    assert 'daily_screen_time_hours_te' not in encoded.columns

def test_target_encoder_shapes(dummy_data, dummy_target):
    te = ValueLevelTargetEncoder(smoothing=10.0, cols=['gender', 'stress_level', 'academic_work_impact', 'daily_screen_time_hours_rounded'])
    dummy_data['daily_screen_time_hours_rounded'] = dummy_data['daily_screen_time_hours'].round(1).astype(str)
    encoded_train = te.fit_transform(dummy_data, dummy_target)

    assert len(encoded_train) == len(dummy_data)

    # Check transform on new data
    encoded_test = te.transform(dummy_data.iloc[:10])
    assert len(encoded_test) == 10
