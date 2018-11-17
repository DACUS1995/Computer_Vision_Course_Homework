import numpy as np
import argparse
import cv2 as cv

import utils

def parse_args():
	parser = argparse.ArgumentParser("Line-Detection")
	parser.add_argument("--file", type=str, default="1.jpg", help="name of the image to process")
	args = parser.parse_args()
	return args

def detect_lines(image):
	edges = extract_edges(image)
	utils.show_image(edges)
	return edges


def extract_edges(image):
	edges = cv.Canny(image, 100, 200, L2gradient=True)
	return edges



def main():
	args = parse_args()

	image = utils.load_image(args.file)
	detect_lines(image)


if __name__ == "__main__":
	main()