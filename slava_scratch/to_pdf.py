# %%
import plotly.io as pio
import time

def convert_json_to_pdf(json_filename, pdf_filename):
    # Load the figure from the JSON file
    fig = pio.read_json(json_filename)
    # Save the figure as a PDF file
    time.sleep(1)
    pio.write_image(fig, pdf_filename, format='pdf')
    print(f"Converted {json_filename} to {pdf_filename}")

# %%
if __name__ == "__main__":
    base_path = "../baselines"
    # Convert the first figure
    convert_json_to_pdf(f"{base_path}/all_plots_2b.json", f"{base_path}/all_plots_2b.pdf")
    # Convert the second figure
    convert_json_to_pdf(f"{base_path}/London_wedding_optimised_metrics_2b.json", f"{base_path}/London_wedding_optimised_metrics_2b.pdf")

    # Uncomment the following lines if you have the 9b data and JSON files
    convert_json_to_pdf(f"{base_path}/all_plots_9b.json", f"{base_path}/all_plots_9b.pdf")
    convert_json_to_pdf(f"{base_path}/London_wedding_optimised_metrics_9b.json", f"{base_path}/London_wedding_optimised_metrics_9b.pdf")

    ##### convert like above but for _surprisingly
    convert_json_to_pdf(f"{base_path}/all_plots_2b_surprisingly.json", f"{base_path}/all_plots_2b_surprisingly.pdf")

    

# %%
