import numpy as np
import pandas as pd
import vectorlab as vl

n_samples = 100
n_entities = 2

ts = np.tile(
    np.arange(n_samples),
    n_entities
)
sin_x = np.sin(ts)
cos_y = np.cos(ts)
entities = np.repeat(
    ['sample_1', 'sample_2'],
    n_samples
)

df = pd.DataFrame()
df['entity'] = entities
df['ts'] = ts
df['sin_x'] = sin_x
df['cos_y'] = cos_y

ts_dataset = vl.data.dataset.TSDataset()
ts_dataset.from_pandas(
    df,
    entity='entity',
    timestamp='ts'
)

ts_dataset.show(show_date=False)

sample_data_2 = ts_dataset['sample_2']

sample_data_2.format_timestamp(step=0.1)
sample_data_2.interpolate(kind='cubic')
sample_data_2.standardize()
sample_data_2.preprocessing(method='minmax')

sample_data_2.show(
    attr_names=['sin_x', 'cos_y'],
    compress=False,
    show_date=True
)

ts_dataset.show(show_date=False)
