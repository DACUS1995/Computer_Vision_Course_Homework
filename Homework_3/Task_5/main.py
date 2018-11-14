import numpy as np

import utils
from Task_4 import apply_opening, apply_closing
from Task_2 import run_skin_hue_threshold

group_counter = 2

def preprocess_image(rgb_image, hsv_image):
	print("--> Preprocessing the input image")
	resulted_image = run_skin_hue_threshold(hsv_image, rgb_image)
	resulted_image = apply_opening(resulted_image)
	resulted_image = apply_closing(resulted_image)
	return resulted_image

def group_blobs(image):
	new_image = np.zeros(image.shape)
	print("---> Running blob Grouping")

	linked = {}

	global group_counter

	# Forward pass
	for x in range(1, image.shape[0] - 1):
		for y in range(1, image.shape[1] - 1):
			if image[x,y] != 0:
				if np.sum(new_image[x-1:x+2, y-1:y+2]) == 0:
					linked[group_counter] = set()
					new_image[x,y] = group_counter
					group_counter = group_counter + 1
				else:
					L = new_image[x-1:x+2, y-1:y+2].flatten()
					L = L[L > 1]
					new_image[x,y] = min(L)
					for label in L:
						linked[label] = linked[label] | set(L)

	for x in range(1, image.shape[0] - 1):
		for y in range(1, image.shape[1] - 1):
			if new_image[x,y] != 0:
				new_image[x,y] = sorted(linked[label])[0]

	return new_image

def compute_centers(image_map):
	print("---> Computing center of blobs")
	centers = []
	global group_counter

	for groups in range(2, group_counter):
		positions = np.where(image_map == groups)
		if positions[0].size > 0:
			x_mean = np.mean(positions[0])
			y_mean = np.mean(positions[1])
			centers.append((x_mean, y_mean))
	return centers

def main():
	rgb_image = utils.load_image("5.jpg")
	hsv_image = utils.rgb_to_hsv(rgb_image)

	resulted_image = preprocess_image(rgb_image, hsv_image)
	utils.show_image(resulted_image)

	grouped_image = group_blobs(resulted_image)
	utils.show_image(grouped_image)

	centers = compute_centers(grouped_image)
	centers = [{"pos_x": center[0], "pos_y": center[1]} for center in centers]

	print(centers)

	drawn = utils.draw_multiple_ellipses(rgb_image, centers)
	utils.show_image(drawn)



if __name__ == "__main__":
	main()