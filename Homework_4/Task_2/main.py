import numpy as np
import argparse
import cv2 as cv
from typing import Dict, Tuple, List

from Task_1 import detect_lines
from Task_2.rectangles import Rectangle_finder
import utils

def parse_args():
	parser = argparse.ArgumentParser("Line-Detection")
	parser.add_argument("--file", type=str, default="1.jpg", help="name of the image to process")
	args = parser.parse_args()
	return args


def main():
	args = parse_args()

	image = utils.load_image(args.file)
	output_image, lines_eq, lines_points, lines_class = detect_lines(image, use_original=True)

	rectangles = Rectangle_finder.search(lines_eq, lines_points, lines_class)
	print(rectangles)
	for points in rectangles:
		print()
		output_image = utils.draw_circle(output_image, *points)

	utils.show_image(output_image)

if __name__ == "__main__":
	main()