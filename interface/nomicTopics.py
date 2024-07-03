from nomic import AtlasDataset
import pandas as pd
import json

dataset = AtlasDataset('siunamimatt/autointerp-v1')
map = dataset.maps[0]

# Assuming map.topics and map.data.df are both pandas DataFrames
topics = map.topics.df  # Ensure this is a DataFrame
data_df = map.data.df

print(topics)
print(data_df)