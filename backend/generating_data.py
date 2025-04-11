import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000
PATH_TO_DATA = '../data'

sample_ads = pd.DataFrame({
    'ad_id': np.arange(n),
    'campaign': np.random.choice(['Brand_Awareness', 'Retargeting', 'Conversions', 'Traffic'], n),
    'ad_spend': np.random.uniform(10, 500, n),
    'impressions': np.random.randint(1000, 50000, n),
    'date': pd.date_range(start='2023-01-01', periods=n, freq='h')
})

click_rate = np.random.uniform(0.01, 0.1, n)
sample_ads['clicks'] = (sample_ads['impressions'] * click_rate).astype(int)
sample_ads['ctr'] = sample_ads['clicks'] / sample_ads['impressions']
sample_ads['conversions'] = np.random.poisson(lam=5, size=n)
sample_ads['cpc'] = sample_ads['ad_spend'] / sample_ads['clicks'].replace(0, 1)

sample_ads.to_csv(f'{PATH_TO_DATA}/meta_data.csv', index=False)
