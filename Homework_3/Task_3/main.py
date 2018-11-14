import numpy as np
import utils
import math

from Task_2 import run_skin_hue_threshold

def dilate(image, kernel_size = 5):
	print("---> Dilating")
	if kernel_size % 2 == 0:
		raise Exception("The kernel size must be an odd number.")

	kernel = np.ones((kernel_size, kernel_size))
	new_image = np.zeros((image.shape[0] + kernel_size, image.shape[1] + kernel_size))
	
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			if image[i, j] == 1:
				new_image[
					i : i + 2 * int(kernel_size / 2) + 1,
					j : j + 2 * int(kernel_size / 2) + 1
				] = kernel
	# utils.show_image(image)
	# utils.show_image(new_image)
	return new_image[
		int(kernel_size / 2) : new_image.shape[0] - int(kernel_size / 2) - 1, 
		int(kernel_size / 2) : new_image.shape[1] - int(kernel_size / 2) - 1
	]

def erode(image, kernel_size = 5):
	print("---> Eroding")
	if kernel_size % 2 == 0:
		raise Exception("The kernel size must be an odd number.")

	new_image = np.zeros(image.shape)	
	
	for i in range(image.shape[0] - kernel_size + 1):
		for j in range(image.shape[1] - kernel_size + 1):
			if np.sum(image[
				i : i + kernel_size, j : j + kernel_size
			]) == kernel_size ** 2:
				new_image[i + int(kernel_size / 2), j + int(kernel_size / 2)] = 1
	# utils.show_image(image)
	# utils.show_image(new_image)
	return new_image

def main():
	rgb_image = utils.load_image("5.jpg")
	hsv_image = utils.rgb_to_hsv(rgb_image)

	resulted_image = run_skin_hue_threshold(hsv_image, rgb_image)

	resulted_image = erode(resulted_image, kernel_size=7)
	resulted_image = dilate(resulted_image, kernel_size=7)
	# utils.show_image(resulted_image)

if __name__ == "__main__":
	main()