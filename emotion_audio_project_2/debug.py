import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
import pyarrow.parquet as pq

parquet_file = 'C:\\Users\\zixua\\Downloads\\session1-00000-of-00001.parquet'
table = pq.read_table(parquet_file)
df = table.to_pandas()

print('Audio column type:', type(df['audio'].iloc[0]))
audio_obj = df['audio'].iloc[0]
print('Audio object:', audio_obj)
if isinstance(audio_obj, dict):
    print('Keys:', list(audio_obj.keys()))
    if 'array' in audio_obj:
        print('Array type:', type(audio_obj['array']))
        print('Array shape:', audio_obj['array'].shape if hasattr(audio_obj['array'], 'shape') else len(audio_obj['array']))
