from nomic import AtlasDataset
import pandas as pd
import json

dataset = AtlasDataset('siunamimatt/autointerp-v1')
map = dataset.maps[0]

# Assuming map.topics and map.data.df are both pandas DataFrames
topics_df = map.topics.df
data_df = map.data.df

# Join the dataframes on 'row_number'
merged_df = pd.merge(topics_df, data_df, on='row_number')

def clean_topic(topic):
    # Remove any occurrences of (2) or similar annotations
    return topic.replace('(2)', '').strip()

def construct_topic_tree(df):
    # Clean the topics
    df['topic_depth_1'] = df['topic_depth_1'].apply(clean_topic)
    df['topic_depth_2'] = df['topic_depth_2'].apply(clean_topic)
    
    # Create a nested dictionary to represent the tree
    topic_tree = {}
    for _, row in df.iterrows():
        depth_1 = row['topic_depth_1']
        depth_2 = row['topic_depth_2']
        data_item = row.to_dict()
        
        if depth_1 not in topic_tree:
            topic_tree[depth_1] = {}
        
        if depth_2 not in topic_tree[depth_1]:
            topic_tree[depth_1][depth_2] = []
        
        topic_tree[depth_1][depth_2].append(data_item)
    
    return topic_tree

def save_topic_tree_to_json(topic_tree, filename):
    with open(filename, 'w') as f:
        json.dump(topic_tree, f, indent=4)

# Construct the topic tree
topic_tree = construct_topic_tree(merged_df)

# Save the topic tree to a JSON file
save_topic_tree_to_json(topic_tree, 'topic_tree.json')
