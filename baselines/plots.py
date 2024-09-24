# %%
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def main():
    # Load the graph data from the JSON file
    with open('graph_data_all_methods.json', 'r') as f:
        graph_data_list = json.load(f)
    
    # Get the unique steering goals
    steering_goals = sorted(set(data['steering_goal_name'] for data in graph_data_list))
    
    # Prepare a mapping from steering goal to its data
    data_by_goal = {}
    for goal in steering_goals:
        data_by_goal[goal] = []
    for data in graph_data_list:
        goal = data['steering_goal_name']
        data_by_goal[goal].append(data)
    
    # Create subplots: 5 columns, as many rows as needed
    num_goals = len(steering_goals)
    cols = 3
    rows = (num_goals + cols - 1) // cols  # Ceiling division to get the number of rows
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=steering_goals)
    
    # Prepare colors for different methods
    method_colors = {
        'ActSteer': 'blue',
        'SAE': 'green',
        'OptimisedSteer': 'red',
    }
    
    # Add data to each subplot
    for idx, goal in enumerate(steering_goals):
        row = idx // cols + 1
        col = idx % cols + 1
        goal_data = data_by_goal[goal]
        
        for method_data in goal_data:
            method = method_data['method']
            scales = method_data['scales']
            product = method_data['product']
            fig.add_trace(
                go.Scatter(
                    x=scales,
                    y=product,
                    mode='lines',
                    name=method,
                    line=dict(color=method_colors.get(method, 'black')),
                    showlegend=(row == 1 and col == 1)  # Show legend only in the first subplot
                ),
                row=row,
                col=col
            )
    
    # Update the layout of the figure
    fig.update_layout(
        height=300 * rows,  # Adjust the height as needed
        width=1200,         # Adjust the width as needed
        title_text='Steering Analysis - Product vs Scale',
        legend_title='Method',
        showlegend=True
    )
    
    # Update axis labels and fonts for all subplots
    fig.update_xaxes(title_text='Scale', row=rows, col=1)
    fig.update_yaxes(title_text='Coherence * Score', row=1, col=1)
    fig.update_annotations(font_size=12)
    
    fig.show()
# %%
if __name__ == "__main__":
    main()

# %%
