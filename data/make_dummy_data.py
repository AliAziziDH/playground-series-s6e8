import pandas as pd
import numpy as np

np.random.seed(42)
N_TRAIN = 1000
N_TEST = 500

def make_data(N):
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

    # Add random NaNs to test robustness
    for col in ['gaming_hours', 'work_study_hours', 'social_media_hours']:
        mask = np.random.rand(N) < 0.1
        df.loc[mask, col] = np.nan

    return df

train = make_data(N_TRAIN)
train['target'] = np.random.randint(0, 2, N_TRAIN)
test = make_data(N_TEST)

train.to_csv('data/train.csv', index=False)
test.to_csv('data/test.csv', index=False)
print("Dummy data created.")
