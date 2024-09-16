# %%

import json
import pandas as pd
import plotly.express as px
import os

def main():
    # Load the data from steering_results.json
    with open('steering_results.json', 'r') as f:
        data = json.load(f)

    # Convert the data into a pandas DataFrame
    df = pd.DataFrame(data)

    # Extract the steering goal name
    df['steering_goal'] = df.apply(
        lambda row: row['steering_goal_name'] if row['steering_goal_name'] != 'Unknown' else os.path.basename(row['path']),
        axis=1
    )

    # Prepare the DataFrame for visualization
    df_summary = df[['steering_goal', 'method', 'max_product', 'scale_at_max']]

    # Create the summary table
    summary_table = df_summary.pivot_table(index='steering_goal', columns='method', values='max_product')

    # Print the summary table
    print("Summary Table:")
    print(summary_table)

    # Save the summary table to a CSV file
    summary_table.to_csv('steering_results_summary.csv')

    # Create an interactive bar plot using Plotly
    fig = px.bar(
        df_summary,
        x='steering_goal',
        y='max_product',
        color='method',
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

    # Save the interactive plot as an HTML file
    fig.write_html('steering_results_plot.html')

    # Show the plot in the default web browser
    fig.show()

# %%
if __name__ == "__main__":
    main()
# %%
