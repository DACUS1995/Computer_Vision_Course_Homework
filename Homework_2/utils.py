import math
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

current_dir_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
argv = sys.argv

def load_image(image_name):
	path_to_image = os.path.join(current_dir_path, "Homework_2", "images", "stop_signs", image_name)
	image_matrix = mpimg.imread(path_to_image)
	return image_matrix


def show_image(image):
	plt.imshow(image)
	plt.show()

if __name__ == "__main__":
	raise Exception("Must be runned as a module.")
