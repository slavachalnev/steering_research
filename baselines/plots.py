# %%
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.express as px

# %%
def main(data_path, is_2b=True, name=""):
    with open(data_path, 'r') as f:
        graph_data_list = json.load(f)

    # Get the unique steering goals and sort them case-insensitively
    steering_goals = sorted(set(data['steering_goal_name'] for data in graph_data_list), key=str.lower)

    # Capitalize the steering goals for display
    capitalized_goals = [goal.capitalize() for goal in steering_goals]

    # Prepare a mapping from steering goal to its data
    data_by_goal = {goal: [] for goal in steering_goals}
    for data in graph_data_list:
        goal = data['steering_goal_name']
        data_by_goal[goal].append(data)

    # Create subplots: 3 columns, as many rows as needed
    num_goals = len(steering_goals)
    cols = 3
    rows = (num_goals + cols - 1) // cols  # Ceiling division to get the number of rows
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=capitalized_goals)

    # Get Plotly's default qualitative color palette
    default_colors = px.colors.qualitative.Plotly

    # Prepare colors for different methods using Plotly's default colors
    method_colors = {
        'CAA': default_colors[0],       # Typically blue
        'SAE': default_colors[1],            # Typically red/orange
        'SAE-TS': default_colors[2], # Typically green
    }

    # Add data to each subplot
    for idx, goal in enumerate(steering_goals):
        row = idx // cols + 1
        col = idx % cols + 1
        goal_data = data_by_goal[goal]

        for method_data in goal_data:
            method = method_data['method']
            if method == 'RotationSteer':
                continue  # Skip RotationSteer
            if method == 'PinverseSteer':
                continue  # Skip PinverseSteer
            
            # Map the method names
            if method == 'ActSteer':
                method = 'CAA'
            elif method == 'OptimisedSteer':
                method = 'SAE-TS'
            
            scales = method_data['scales']
            product = method_data['product']
            fig.add_trace(
                go.Scatter(
                    x=scales,
                    y=product,
                    mode='lines',
                    name=method,
                    line=dict(color=method_colors.get(method, default_colors[0])),  # Use Plotly default colors
                    showlegend=(row == 1 and col == 1)  # Show legend only in the first subplot
                ),
                row=row,
                col=col
            )

    # Update the layout of the figure
    fig.update_layout(
        height=300 * rows,  # Adjust the height as needed
        width=1200,         # Adjust the width as needed
        # title_text='Steering Analysis - Product vs Scale',
        legend_title='Method',
        showlegend=True,
        legend_font_size=14
    )

    # Update axis labels and fonts for all subplots
    fig.update_xaxes(title_text='Scale', row=rows, col=1)
    fig.update_xaxes(title_text='Scale', row=rows, col=2)
    fig.update_xaxes(title_text='Scale', row=rows, col=3)
    for row in range(1, rows + 1):
        fig.update_yaxes(title_text='Coherence * Score', row=row, col=1)

    # fig.update_yaxes(title_text='Coherence * Score', row=1, col=1)
    fig.update_annotations(font_size=18)

    # Save the figure as a JSON file
    if is_2b:
        output_filename = f"all_plots_2b{name}.json"
    else:
        output_filename = f"all_plots_9b{name}.json"
    pio.write_json(fig, output_filename)
    print(f"Figure saved as {output_filename}")

    # Optionally, still show the figure in the notebook
    fig.show()


# %%
def plot_specific_goals(data_path, is_2b=True, name=""):
    # Define the specific goals to plot
    goals_to_plot = ["London", "wedding"]

    # Load the graph data from the JSON file
    with open(data_path, 'r') as f:
        graph_data_list = json.load(f)

    # Filter the data for the specified goals and the 'OptimisedSteer' method
    graph_data_list = [
        data for data in graph_data_list
        if data['steering_goal_name'] in goals_to_plot and data['method'] == 'OptimisedSteer'
    ]

    # Get the unique steering goals (after filtering) and sort them case-insensitively
    steering_goals = sorted(set(data['steering_goal_name'] for data in graph_data_list), key=str.lower)

    # Capitalize the steering goals for display
    capitalized_goals = [goal.capitalize() for goal in steering_goals]

    # Prepare a mapping from steering goal to its data
    data_by_goal = {goal: [] for goal in steering_goals}
    for data in graph_data_list:
        goal = data['steering_goal_name']
        data_by_goal[goal].append(data)

    # Create subplots: 2 columns (side by side), 1 row
    cols = 2
    rows = 1
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=capitalized_goals)

    # Line styles for different metrics
    metric_styles = {
        'product': {'dash': 'solid', 'width': 3},
        'avg_coherence': {'dash': 'dot', 'width': 2},
        'avg_score': {'dash': 'dash', 'width': 2},
    }

    # Get default Plotly colors
    default_colors = px.colors.qualitative.Plotly
    # Assign colors to metrics using default colors
    metric_colors = {
        'product': default_colors[2],       # Third color
        'avg_coherence': default_colors[1], # Second color
        'avg_score': default_colors[0],     # First color
    }

    legend_names = {
        'product': 'behavioral*coherence',
        'avg_coherence': 'coherence',
        'avg_score': 'behavioral'
    }

    # Add data to each subplot
    for idx, goal in enumerate(steering_goals):
        row = 1
        col = idx + 1
        goal_data = data_by_goal[goal][0]  # Only one method ('OptimisedSteer') per goal

        # Limit scales to 180
        scales = [s for s in goal_data['scales'] if s <= 180]
        # Find indices where scales are <= 180
        valid_indices = [i for i, s in enumerate(goal_data['scales']) if s <= 180]

        # Get the corresponding metric values up to scale 180
        metrics = {
            'product': [goal_data['product'][i] for i in valid_indices],
            'avg_coherence': [goal_data['avg_coherence'][i] for i in valid_indices],
            'avg_score': [goal_data['avg_score'][i] for i in valid_indices],
        }

        for metric_name, metric_values in metrics.items():
            style = metric_styles.get(metric_name, {})
            color = metric_colors.get(metric_name, 'black')
            fig.add_trace(
                go.Scatter(
                    x=scales,
                    y=metric_values,
                    mode='lines',
                    name=legend_names[metric_name] if idx == 0 else None,
                    line=dict(
                        color=color,
                        dash=style.get('dash', 'solid'),
                        width=style.get('width', 2),
                    ),
                    showlegend=(idx == 0)
                ),
                row=row,
                col=col
            )

    # Update the layout of the figure
    fig.update_layout(
        height=450,  # Reduced height for less tall plots
        width=1200,  # Adjusted width for side by side plots
        # title_text='Targeted Steering Metrics for London and Wedding',
        legend_title='Metric',
        showlegend=True,
        legend_font_size=16
    )

    # Update axis labels and fonts for all subplots
    for c in range(1, cols + 1):
        fig.update_xaxes(title_text='Scale', row=1, col=c, range=[0, 180], tickfont=dict(size=14), title_font=dict(size=16))
        fig.update_yaxes(title_text='Value', row=1, col=c, range=[0, 1], tickfont=dict(size=14), title_font=dict(size=16))
    fig.update_annotations(font_size=18)

    # Save the figure as a JSON file
    if is_2b:
        output_filename = f"London_wedding_optimised_metrics_2b{name}.json"
    else:
        output_filename = f"London_wedding_optimised_metrics_9b{name}.json"
    pio.write_json(fig, output_filename)
    print(f"Figure saved as {output_filename}")

    # Optionally, show the figure in the notebook
    fig.show()


# %%
if __name__ == "__main__":
    name=""
    # name="_surprisingly"

    main(data_path=f"graph_data_all_methods_gemma-2-2b{name}.json", is_2b=True, name=name)
    plot_specific_goals(data_path=f"graph_data_all_methods_gemma-2-2b{name}.json", is_2b=True, name=name)

    # main(data_path="graph_data_all_methods_gemma-2-9b.json", is_2b=False, name=name)
    # plot_specific_goals(data_path="graph_data_all_methods_gemma-2-9b.json", is_2b=False, name=name)

# %%
