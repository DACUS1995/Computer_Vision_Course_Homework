
import math
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from gaussian_filter import GaussianFilter
from box_filter import BoxFilter

current_dir_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
argv = sys.argv

def load_image(image_name):
	path_to_image = os.path.join(current_dir_path, "images", "stop_signs", image_name)
	image_matrix = mpimg.imread(path_to_image)
	return image_matrix


def show_image(image):
	plt.imshow(image)
	plt.show()


def run_gaussian_filter(image_matrix):
	# Gaussian Filter
	gauss_filter = GaussianFilter(size=11)
	filter = gauss_filter.generate_filter(sigma=5)
	print(filter.sum())
	
	for line in filter:
  		print (["%.3f" % x for x in line])

	new_image = gauss_filter.filter_image(image_matrix)
	show_image(new_image)


def run_box_filter(image_matrix):
	# Box Filter
	box_filter = BoxFilter(5)
	filter = box_filter.generate_filter()

	for line in filter:
  		print (["%.3f" % x for x in line])

	new_image = box_filter.filter_image(image_matrix)
	show_image(new_image)


def main():
	image_matrix = load_image("stop_sign_01.jpg")
	show_image(image_matrix)

	if argv[1] == "gaussian":
		run_gaussian_filter(image_matrix)
	
	if argv[1] == "box":
		run_box_filter(image_matrix)

if __name__ == "__main__":
	main()
