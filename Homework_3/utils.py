import math
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as colors

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

if __name__ == "__main__":
	raise Exception("Must not be runned as a module.")
