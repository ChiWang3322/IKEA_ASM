import matplotlib.pyplot as plt
import numpy as np

# generate some random data
data = np.random.rand(1, 5)

# define the categories
categories = ['Category 1']

# define the colors for each category
colors = ['red', 'blue', 'green', 'orange', 'c', 'm', 'y', 'k']

# create the plot
fig, ax = plt.subplots()

# loop through each category and plot a segment for each value
for i, cat in enumerate(categories):
    left = 0
    for j, val in enumerate(data[i]):
        ax.barh(i, val, left=left, color=colors[j], edgecolor='white', height=0.1)
        left += val

# add labels and title
ax.set_xlabel('Values')
ax.set_ylabel('Categories')
ax.set_title('Horizontal Segment Bar Plot')

# add legend
legend_handles = [plt.Rectangle((0,0),1,1, color=colors[i], edgecolor='white') for i in range(len(colors))]
ax.legend(legend_handles, ['Value 1', 'Value 2', 'Value 3', 'Value 4', 'Value 5'])

# show the plot
plt.show()
