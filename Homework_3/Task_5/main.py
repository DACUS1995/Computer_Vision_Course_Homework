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
					linked[group_counter] = set([group_counter])
					new_image[x,y] = group_counter
					group_counter = group_counter + 1
				else:
					L = new_image[x-1:x+2, y-1:y+2].flatten()
					L = L[L > 1]
					new_image[x,y] = min(L)
					for label in L:
						linked[label] = linked[label] | set(L)
	utils.show_image(new_image)

	for x in range(1, image.shape[0] - 1):
		for y in range(1, image.shape[1] - 1):
			if new_image[x,y] != 0:
				new_image[x,y] = sorted(linked[new_image[x,y]])[0]

	class_list = np.array(list(set(new_image.flatten())))
	class_list = class_list[class_list != 0]
	return new_image, class_list


def compute_centers(image_map, classes, faces):
	print("---> Computing center of blobs")

	for group in range(classes):
		positions = np.where(image_map == group)

		if positions[0].size > 0 and (group in faces):
			x_mean = np.mean(positions[0])
			y_mean = np.mean(positions[1])
			faces[group]["pos_x"] = x_mean
			faces[group]["pos_y"] = y_mean
	return faces


def compute_orientation(image):
	print("---> Computing orientation")
	y, x = np.nonzero(image)

	x = x - np.mean(x)
	y = y - np.mean(y)
	coords = np.vstack((x ,y))
	cov = np.cov(coords)
	print(cov)
	evals, evecs = np.linalg.eig(cov)

	sort_indices = np.argsort(evals)[::-1]
	x_v1, y_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
	x_v2, y_v2 = evecs[:, sort_indices[1]]

	theta = np.tanh((x_v1)/(y_v1))  
	return theta

def main():
	rgb_image = utils.load_image("5.jpg")
	hsv_image = utils.rgb_to_hsv(rgb_image)

	resulted_image = preprocess_image(rgb_image, hsv_image)
	utils.show_image(resulted_image)

	grouped_image, class_list = group_blobs(resulted_image)
	utils.show_image(grouped_image)

	faces = {}

	for el in class_list:
		pos = np.where(grouped_image == el)
		max_x = max(pos[0])
		min_y = min(pos[1])
		min_x = min(pos[0])
		max_y = max(pos[1])

		# Applying proportion threshold
		height = max_x - min_x
		width = max_y - min_y

		if height / width <  2:
			faces[el] = {}
			faces[el]["size_x"] = height
			faces[el]["size_y"] = width

			faces[el]["angle"] = compute_orientation(resulted_image[min_x:max_x, min_y:max_y])

	faces = compute_centers(grouped_image, class_list, faces)

	drawn = utils.draw_multiple_ellipses(rgb_image, faces)
	utils.show_image(drawn)



if __name__ == "__main__":
	main()