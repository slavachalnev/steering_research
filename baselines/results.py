# %%
import json
import pandas as pd
import plotly.express as px
import os

def main(results_json):
    with open(results_json, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df['steering_goal'] = df.apply(
        lambda row: row['steering_goal_name'] if row['steering_goal_name'] != 'Unknown' else os.path.basename(row['path']),
        axis=1
    )
    df_summary = df[['steering_goal', 'method', 'max_product', 'scale_at_max']]
    summary_table = df_summary.pivot_table(index='steering_goal', columns='method', values='max_product')
    print("Summary Table:")
    print(summary_table)
    summary_table.to_csv('steering_results_summary.csv')
    
    # Define a custom color map using Plotly-like colors
    color_map = {
        'ActSteer': '#636EFA',  # Plotly's default blue
        'SAE': '#EF553B',        # Plotly's default red
        'OptimisedSteer': '#00CC96'              # Plotly's default green
    }
    
    fig = px.bar(
        df_summary,
        x='steering_goal',
        y='max_product',
        color='method',
        color_discrete_map=color_map,  # Apply the custom color map
        barmode='group',
        title='Max Product by Steering Goal and Method',
        labels={'max_product': 'Max Product', 'steering_goal': 'Steering Goal'},
        hover_data=['scale_at_max']
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        legend_title_text='Method',
        xaxis_title='Steering Goal',
        yaxis_title='Max Product'
    )
    fig.show()
# %%
if __name__ == "__main__":
    main('steering_results.json')
    # main('steering_results_london.json')
# %%
