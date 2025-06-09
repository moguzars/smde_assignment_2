import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create the DataFrame from the provided data
data = {
    'A': [0, 1, 0, 1, 0, 1, 0, 1],
    'B': [0, 0, 1, 1, 0, 0, 1, 1],
    'C': [0, 0, 0, 0, 1, 1, 1, 1],
    'Response': [309286300,298821700,578604700,558786700,512332100,489545500,573589000,540131500],
}
df = pd.DataFrame(data)

# Create a 'Run' column for plotting purposes, combining the factor levels
df['Run'] = df.apply(lambda row: f"A={row['A']}, B={row['B']}, C={row['C']}", axis=1)

# Set the style of the plot
sns.set_style("whitegrid")

# Create the plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Run', y='Response', data=df, s=100 , palette='deep')
sns.lineplot(x='Run', y='Response', data=df, color='gray', linestyle='--', alpha=0.5)

# Add labels and title
plt.xlabel("Factor Combinations (A, B, C)", fontsize=12)
plt.ylabel("Profit", fontsize=12)
plt.title("Profit Across Factor Combinations", fontsize=14)
plt.xticks(rotation=45, ha='right') # Rotate x-axis labels for better readability
plt.tight_layout() # Adjust layout to prevent labels from overlapping
plt.show()
