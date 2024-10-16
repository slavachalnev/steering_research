# %%
import json
import pandas as pd
import plotly.express as px
import os



def main(results_json, is_2b=True):
    with open(results_json, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df['steering_goal'] = df.apply(
        lambda row: row['steering_goal_name'] if row['steering_goal_name'] != 'Unknown' else os.path.basename(row['path']),
        axis=1
    )
    # Capitalize the steering goals
    df['steering_goal'] = df['steering_goal'].str.capitalize()
    
    df_summary = df[['steering_goal', 'method', 'max_product', 'scale_at_max']]
    summary_table = df_summary.pivot_table(index='steering_goal', columns='method', values='max_product')
    
    # Sort the summary table alphabetically by index (steering_goal), still ignoring case for sorting
    summary_table_sorted = summary_table.sort_index(key=lambda x: x.str.lower())
    
    # Round the sorted summary table to 4 decimal places
    summary_table_rounded = summary_table_sorted.round(4)
    
    print("Summary Table (Alphabetically Ordered, Capitalized):")
    print(summary_table_rounded)
    
    # Optionally, save the rounded and sorted table to CSV
    summary_table_rounded.to_csv('steering_results_summary_sorted_capitalized.csv')
    
    # Define a custom color map using Plotly-like colors
    color_map = {
        'CAA': '#636EFA',  # Plotly's default blue
        'SAE': '#EF553B',        # Plotly's default red
        'SAE-TS': '#00CC96',  # Plotly's default green
        'RotationSteer': '#AB63FA'  # Plotly's default purple
    }

    # Update method names in the DataFrame
    method_name_map = {
        'ActSteer': 'CAA',
        'OptimisedSteer': 'SAE-TS'
    }
    df_summary['method'] = df_summary['method'].replace(method_name_map)
    
    # Sort df_summary by steering_goal for consistent ordering in the plot, ignoring case
    df_summary_sorted = df_summary.sort_values('steering_goal', key=lambda x: x.str.lower())
    
    fig = px.bar(
        df_summary_sorted,
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
    
    # Save the plot as a PDF file
    if is_2b:
        output_filename = "steering_results_plot_2b.pdf"
    else:
        output_filename = "steering_results_plot_9b.pdf"
    fig.write_image(output_filename)
    print(f"Plot saved as {output_filename}")
    
    # Display the plot
    fig.show()
# %%
if __name__ == "__main__":
    main('steering_results_gemma-2-2b.json', is_2b=True)
    # main('steering_results_london.json')
# %%
