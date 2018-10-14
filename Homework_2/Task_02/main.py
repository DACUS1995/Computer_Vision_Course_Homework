
import math
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from gaussian_filter import GaussianFilter

current_dir_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))


def load_image(image_name):
	path_to_image = os.path.join(current_dir_path, "images", "stop_signs", image_name)
	image_matrix = mpimg.imread(path_to_image)
	return image_matrix

def show_image(image):
	plt.imshow(image)
	plt.show()

def main():
	image_matrix = load_image("stop_sign_01.jpg")
	# show_image(image_matrix)

	# Gaussian Filter
	gauss_filter = GaussianFilter(5)
	filter = gauss_filter.generate_filter()
	for line in filter:
  		print (["%.3f" % x for x in line])
	new_image = gauss_filter.filter_image(image_matrix)

	show_image(new_image)


if __name__ == "__main__":
	main()
