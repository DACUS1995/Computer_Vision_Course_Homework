import math
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as colors
import cv2
import math as math

current_dir_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
argv = sys.argv

def load_image(image_name):
	path_to_image = os.path.join(current_dir_path, "Homework_3", "Images", image_name)
	image_matrix = mpimg.imread(path_to_image)
	return image_matrix


def show_image(image):
	if len(image.shape) == 3:
		plt.imshow(image)
	else:
		plt.imshow(image, cmap="gray")
	plt.show()

def rgb_to_hsv(image):
	new_image = colors.rgb_to_hsv(image)
	return new_image

def show_hsv_image(image):
	show_hsv_image(colors.hsv_to_rgb(image))


def draw_multiple_ellipses(image, details):
	for i in range(len(details)):
		prop = details[i]
		if not "size_x" in prop:
			prop["size_x"] = None

		if not "size_y" in prop:
			prop["size_y"] = None

		if not "angle" in prop:
			prop["angle"] = None

		image = draw_ellipse(
			image, 
			math.ceil(prop["pos_y"]),
			math.ceil(prop["pos_x"])
		)

	return image


def draw_ellipse(image, pos_x=300, pos_y=200, size_x=100, size_y=50, angle=90):
	new_image = np.copy(image)

	if pos_x == None or pos_y == None:
		raise Exception("The x and y position must be specified.")

	center = (pos_x, pos_y)
	axes = (size_x, size_y)

	print(center, axes)

	cv2.ellipse(new_image, center, axes, angle, 0 , 360, (255,0,0), 2)
	return new_image

if __name__ == "__main__":
	raise Exception("Must not be runned as a module.")
