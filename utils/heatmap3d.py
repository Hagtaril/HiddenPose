# 3D Heatmap in Python using matplotlib

# to make plot interactive

# importing required libraries
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from pylab import *
def plotHeatmap3d(x,y,z,data):
	# creating a dummy dataset
	colo = data

	# creating figures
	fig = plt.figure(figsize=(10, 10))
	ax = fig.add_subplot(111, projection='3d')

	# setting color bar
	color_map = cm.ScalarMappable(cmap=cm.Greens_r)
	color_map.set_array(colo)

	# creating the heatmap
	img = ax.scatter(x, y, z, marker='s',
					s=200, color='green')
	plt.colorbar(color_map)

	# adding title and labels
	ax.set_title("3D Heatmap")
	ax.set_xlabel('X-axis')
	ax.set_ylabel('Y-axis')
	ax.set_zlabel('Z-axis')

	# displaying plot
	plt.show()

if __name__ == '__main__':
	x = np.random.randint(low=100, high=500, size=(1000,))
	y = np.random.randint(low=300, high=500, size=(1000,))
	z = np.random.randint(low=200, high=500, size=(1000,))
	colo = [x + y + z]
	plotHeatmap3d(x,y,z,colo)
