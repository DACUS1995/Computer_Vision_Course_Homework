import numpy as np
import argparse
import cv2 as cv

import utils
from Task_1.ransac import RANSAC

def parse_args():
	parser = argparse.ArgumentParser("Line-Detection")
	parser.add_argument("--file", type=str, default="1.jpg", help="name of the image to process")
	args = parser.parse_args()
	return args

def detect_lines(image, use_original=False):
	original_image = np.copy(image) if use_original else None
	edges = extract_edges(image)
	image, labeled_image, num_labels = find_connected_components(edges)

	# utils.show_image(labeled_image)
	# utils.show_image(image)

	output_image = None
	lines_eq_all = [] 
	lines_points_all = [] 
	lines_class_all = []

	# For each label find the lines the belong to the component
	for label in range(1, num_labels):
		output_image, lines_eq, lines_points, lines_class = RANSAC.search(labeled_image, label, output_image=output_image, original_image=original_image)
		lines_eq_all.extend(lines_eq)
		lines_points_all.extend(lines_points)
		lines_class_all.extend(lines_class)

	return output_image, lines_eq_all, lines_points_all, lines_class_all


def extract_edges(image):
	edges = cv.Canny(image, 100, 200, L2gradient=True)
	# edges = edges / 255
	return edges

def find_connected_components(image):
	num_labels, labeled_image = cv.connectedComponents(image, connectivity=8)
	print("---> Found [", num_labels, "] connected componenets.")
	return image, labeled_image, num_labels


def main():
	args = parse_args()

	image = utils.load_image(args.file)
	output_image, _, _ = detect_lines(image)
	utils.show_image(output_image)


if __name__ == "__main__":
	main()