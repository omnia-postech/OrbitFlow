import matplotlib.pyplot as plt
import numpy as np

# Data labels for the three groups
categories = ['1:1', '2:1', '3:1']

# Sample data for demonstration.
# Replace these lists with your actual data.
layer_traversal_time = [371, 235, 185]   # Example values for layer traversal time
kv_cache_memory      = [8, 11, 12]     # Example values for KV cache memory

# Create an array for the positions of each group on the x-axis
x = np.arange(len(categories))
# Set the width of each bar
width = 0.35

# Create the main figure and the first y-axis (for Layer Traversal Time)
fig, ax1 = plt.subplots()

# Plot the Layer Traversal Time bars on ax1 (shifted left)
rects1 = ax1.bar(x - width/2, layer_traversal_time, width, label='Layer Traversal Time (ms)')
ax1.set_ylabel('Token Generation Time (ms)', color='black')
ax1.tick_params(axis='y', labelcolor='black')

# Create the second y-axis for KV Cache Memory using twinx()
ax2 = ax1.twinx()
# Plot the KV Cache Memory bars on ax2 (shifted right)
rects2 = ax2.bar(x + width/2, kv_cache_memory, width, label='KV Cache Memory (GB)', color='orange')
ax2.set_ylabel('KV Cache Memory (GB)', color='black')
ax1.set_ylim(0, max(layer_traversal_time) * 1.2)
ax2.tick_params(axis='y', labelcolor='black')

# Set the x-axis labels and title on the primary axis
ax1.set_xticks(x)
ax2.set_ylim(0, max(kv_cache_memory) * 1.2)
ax1.set_xticklabels(categories)
ax1.set_title("GPU vs CPU KV Cache Ratio")

# Combine legends from both axes for a unified legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

# Optionally, annotate the bars with their values
def autolabel(rects, axis):
    """Attach a text label above each bar displaying its height."""
    for rect in rects:
        height = rect.get_height()
        axis.annotate(f'{height}',
                      xy=(rect.get_x() + rect.get_width() / 2, height),
                      xytext=(0, 3),  # 3 points vertical offset
                      textcoords="offset points",
                      ha='center', va='bottom')

autolabel(rects1, ax1)
autolabel(rects2, ax2)

fig.tight_layout()  # Adjust layout for neatness
plt.savefig("multi-prefetch-memtime.png")