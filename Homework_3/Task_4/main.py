import numpy as np

import utils
from Task_3 import dilate, erode
from Task_2 import run_skin_hue_threshold

def apply_opening(image):
	print("--> Apply opening transformation")
	# First remove noise
	resulted_image = erode(image, kernel_size=7)
	resulted_image = dilate(resulted_image, kernel_size=7)
	return resulted_image

def apply_closing(image):
	print("--> Apply closing transformation")
	# Fill the goles
	resulted_image = dilate(image, kernel_size=7)
	resulted_image = erode(resulted_image, kernel_size=7)
	return resulted_image

def main():
	rgb_image = utils.load_image("7.jpg")
	hsv_image = utils.rgb_to_hsv(rgb_image)

	resulted_image = run_skin_hue_threshold(hsv_image, rgb_image)
	resulted_image = apply_opening(resulted_image)
	resulted_image = apply_closing(resulted_image)

	# utils.show_image(resulted_image)

	rgb_image.setflags(write=True)
	rgb_image[resulted_image == 0] = 0
	# utils.show_image(rgb_image)

if __name__ == "__main__":
	main()