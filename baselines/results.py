# %%
import json
import pandas as pd
import plotly.express as px
import os

def main(results_json, is_2b=True, name=""):
    with open(results_json, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df['steering_goal'] = df.apply(
        lambda row: row['steering_goal_name'] if row['steering_goal_name'] != 'Unknown' else os.path.basename(row['path']),
        axis=1
    )
    # Capitalize the steering goals
    df['steering_goal'] = df['steering_goal'].str.capitalize()
    
    # Update method names in the DataFrame before creating the pivot table
    method_name_map = {
        'ActSteer': 'CAA',
        'OptimisedSteer': 'SAE-TS'
    }
    df['method'] = df['method'].replace(method_name_map)
    
    df_summary = df[['steering_goal', 'method', 'max_product', 'scale_at_max']]
    summary_table = df_summary.pivot_table(index='steering_goal', columns='method', values='max_product')
    
    # Reorder the columns as per your specification
    desired_order = ['CAA', 'SAE', 'SAE-TS', 'PinverseSteer', 'RotationSteer']
    # Retain only the columns that are present in the DataFrame
    existing_columns = [col for col in desired_order if col in summary_table.columns]
    summary_table = summary_table[existing_columns]
    
    # Sort the summary table alphabetically by index (steering_goal), ignoring case
    summary_table_sorted = summary_table.sort_index(key=lambda x: x.str.lower())
    
    # Round the sorted summary table to 4 decimal places
    summary_table_rounded = summary_table_sorted.round(4)
    
    print("Summary Table (Alphabetically Ordered, Capitalized):")
    print(summary_table_rounded)
    
    # Calculate the average scores for each method (column)
    average_scores = summary_table_rounded.mean()
    print("\nAverage Scores for Each Method:")
    print(average_scores)
    
    # Optionally, save the rounded and sorted table to CSV
    summary_table_rounded.to_csv('steering_results_summary_sorted_capitalized.csv')
    
    # Define a custom color map using Plotly-like colors
    color_map = {
        'CAA': '#636EFA',        # Plotly's default blue
        'SAE': '#EF553B',        # Plotly's default red
        'SAE-TS': '#00CC96',     # Plotly's default green
        'PinverseSteer': '#FFA15A',  # Additional color
        'RotationSteer': '#AB63FA'   # Plotly's default purple
    }
    
    # Sort df_summary by steering_goal for consistent ordering in the plot, ignoring case
    df_summary_sorted = df_summary.sort_values('steering_goal', key=lambda x: x.str.lower())
    
    fig = px.bar(
        df_summary_sorted,
        x='steering_goal',
        y='max_product',
        color='method',
        color_discrete_map=color_map,  # Apply the custom color map
        category_orders={'method': desired_order},  # Set the order of methods in the plot
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
        output_filename = f"steering_results_plot_2b{name}.pdf"
    else:
        output_filename = f"steering_results_plot_9b{name}.pdf"
    fig.write_image(output_filename)
    print(f"Plot saved as {output_filename}")
    
    # Display the plot
    fig.show()
# %%
if __name__ == "__main__":
    name=""
    # name="_surprisingly"
    main(f'steering_results_gemma-2-2b{name}.json', is_2b=True, name=name)
    # main('steering_results_london.json')
# %%
