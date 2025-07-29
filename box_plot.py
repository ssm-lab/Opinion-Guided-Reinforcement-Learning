import matplotlib.pyplot as plt
import numpy as np

# Create some synthetic data
np.random.seed(0)
data = [
    np.random.normal(100, 10, 200),  # Group 1
    np.random.normal(90, 20, 200),   # Group 2
    np.random.normal(80, 30, 200)    # Group 3
]

# Create a box plot
plt.boxplot(data, labels=['Group 1', 'Group 2', 'Group 3'])

# Add title and axis labels
plt.title('Box Plot Example')
plt.ylabel('Values')
plt.xlabel('Groups')

# Show the plot
plt.show()