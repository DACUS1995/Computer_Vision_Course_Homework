import utils

from Task_03.sobel_filter import SobelFilter
from Task_03.gaussian_filter import GaussianFilter

def run_edge_tracing_with_histeresis(image):
	print("---> Running Tracing edges with hysteresis")

	return image

def run_double_threshold(image):
	print("---> Running Threshold")

	return image

def run_suppression(image, angle_map):
	print("---> Running Suppression")
	
	for x in range(1, image.shape[0] - 1):
		for y in range(1, image.shape[1] - 1):

			# Horizontal
			if (angle_map[x][y] > 157.5 or angle_map[x][y] < 22.5) and (image[x][y][0] <= image[x][y-1][0] or image[x][y][0] <= image[x][y+1][0]):
				image[x][y] = 0
			# /
			if (angle_map[x][y] >= 22.5 and angle_map[x][y] < 67.5) and (image[x][y][0] <= image[x-1][y+1][0] or image[x][y][0] <= image[x+1][y-1][0]):
				image[x][y] = 0

			# Vertical
			if (angle_map[x][y] >= 67.5 and angle_map[x][y] < 112.5) and (image[x][y][0] <= image[x-1][y][0] or image[x][y][0] <= image[x+1][y][0]):
				image[x][y] = 0
			# \
			if (angle_map[x][y] >= 112.5 and angle_map[x][y] < 157.5) and (image[x][y][0] <= image[x-1][y-1][0] or image[x][y][0] <= image[x+1][y+1][0]):
				image[x][y] = 0

	return image

def compute_gradients(image):
	print("---> Running Sobel Filter")
	sobel_filter = SobelFilter()
	image, angle_map = sobel_filter.filter_image(image)

	return image, angle_map


def apply_gaussian_filter(image):
	print("---> Running Gaussian Filter")
	gaussian_filter = GaussianFilter(3)

	return image


def run_edge_detector(image):
	image = apply_gaussian_filter(image)
	image, angle_map = compute_gradients(image)
	image = run_suppression(image, angle_map)
	image = run_double_threshold(image)
	image = run_edge_tracing_with_histeresis(image)

	return image

def main():
	image = utils.load_image(image_name = "stop_sign_01.jpg")
	image = run_edge_detector(image)

	utils.show_image(image)

if __name__ == "__main__":
	main()