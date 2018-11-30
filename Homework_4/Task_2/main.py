import numpy as np
import argparse
import cv2 as cv

from Task_1 import detect_lines
import utils

def parse_args():
	parser = argparse.ArgumentParser("Line-Detection")
	parser.add_argument("--file", type=str, default="1.jpg", help="name of the image to process")
	args = parser.parse_args()
	return args


def main():
	args = parse_args()

	image = utils.load_image(args.file)
	output_image = detect_lines(image)
	utils.show_image(output_image)

if __name__ == "__main__":
	main()