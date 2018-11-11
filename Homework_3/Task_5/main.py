import numpy as np

import utils
from Task_4 import apply_opening, apply_closing
from Task_2 import run_skin_hue_threshold

def preprocess_image(rgb_image, hsv_image):
	print("--> Preprocessing the input image")
	resulted_image = run_skin_hue_threshold(hsv_image, rgb_image)
	resulted_image = apply_opening(resulted_image)
	resulted_image = apply_closing(resulted_image)
	return resulted_image


def main():
	rgb_image = utils.load_image("5.jpg")
	hsv_image = utils.rgb_to_hsv(rgb_image)

	draw = utils.draw_ellipse(rgb_image)
	utils.show_image(draw)

	resulted_image = preprocess_image(rgb_image, hsv_image)
	utils.show_image(resulted_image)

if __name__ == "__main__":
	main()