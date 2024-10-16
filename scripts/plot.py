import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the CSV file
df = pd.read_csv("agg_results.csv")

# Init
plt.figure(figsize=(12, 6))

# Create a line plot for the uniform policy
uniform_df = df[df["name"].str.startswith("uniform")]

# Extract the correct x-values (the numbers in the 'name' column)
x_values = uniform_df["name"].str.extract(r"(\d+)", expand=False).astype(int)

# Plot the uniform policy with correct x-values
line_color = "C4"  # You can choose any color you like for uniform
plt.plot(
    x_values,
    -uniform_df["avg_vehicle_disutility"],
    label="uniform",
    color=line_color,
    linewidth=2,
)

# Find the max value for uniform policy and plot a marker
max_index = np.argmax(-uniform_df["avg_vehicle_disutility"])
max_x_value = x_values.iloc[max_index]  # Get the x-value corresponding to the max index
plt.scatter(
    max_x_value,
    -uniform_df["avg_vehicle_disutility"].iloc[max_index],
    marker="*",
    s=200,
    color=line_color,
    label=f"uniform_{max_x_value}",
    edgecolor="black",
)

# Add text annotation for the y-value of the best uniform policy (top-left)
plt.text(
    max_x_value,
    -uniform_df["avg_vehicle_disutility"].iloc[max_index],
    f"{-uniform_df['avg_vehicle_disutility'].iloc[max_index]:.0f}",
    fontsize=12,
    ha="right",
    va="bottom",
    color="lightgray",
)  # Changed color to light gray

# Add markers for the other policies and annotate them
other_policies_df = df[~df["name"].str.startswith("uniform")]
markers = ["o", "s", "^"]
for i, policy in enumerate(other_policies_df["name"]):
    x_value = int(policy.split("_")[-1])
    y_value = -other_policies_df.loc[
        other_policies_df["name"] == policy, "avg_vehicle_disutility"
    ].values[0]
    plt.scatter(
        [x_value],
        [y_value],
        label=policy,
        marker=markers[i],
        s=100,
        alpha=0.8,
        edgecolor="black",
    )

    # Add text for the y-values at each marker (top-left)
    plt.text(
        x_value,
        y_value,
        f"{y_value:.0f}",
        fontsize=12,
        ha="right",
        va="bottom",
        color="lightgray",
    )  # Changed color to light gray

# Add a legend and labels with light font color
plt.legend(fontsize=10, framealpha=0.9, edgecolor="lightgray")  # Legend border color
plt.xlabel(
    "Green Phase Time Interval (s)", fontsize=12, color="lightgray"
)  # Light label color
plt.ylabel(
    "Average Vehicle Disutility (higher is better)", fontsize=12, color="lightgray"
)  # Light label color
plt.title(
    "Policy Comparison", fontsize=14, weight="bold", color="lightgray"
)  # Light title color

# Set the x-axis tick locations and labels
plt.xticks(x_values, x_values, fontsize=10, color="lightgray")  # Light tick color

# Set the y-axis tick locations and labels
plt.yticks(fontsize=10, color="lightgray")  # Set y-tick color to light gray

# Use a log scale for the y-axis
plt.yscale("symlog")
plt.ylim(None, -1e2)

# Beautifying the plot with light grid lines
plt.grid(
    True, which="both", linestyle="--", linewidth=0.7, alpha=0.7, color="lightgray"
)  # Light grid color

# Make the background transparent
plt.gca().set_facecolor("none")  # Set the axes facecolor to transparent
plt.gcf().patch.set_alpha(0.0)  # Set the figure background to transparent

# Set the axes border color
plt.gca().spines["top"].set_color("lightgray")
plt.gca().spines["right"].set_color("lightgray")
plt.gca().spines["left"].set_color("lightgray")
plt.gca().spines["bottom"].set_color("lightgray")

# Use tight layout
plt.tight_layout()

# Show the plot
# plt.show()

# Optionally, save the figure with a transparent background
plt.savefig(
    ".\\images\\results_graph.png", transparent=True
)  # Uncomment to save the plot
