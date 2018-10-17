import utils
from ..Task_02.gaussian_filter import GaussianFilter

def run_edge_tracing_with_histeresis(image):
	print("---> Running Tracing edges with hysteresis")

	return image

def run_double_threshold(image):
	print("---> Running Threshold")

	return image

def run_suppression(image):
	print("---> Running Suppression")

	return image

def compute_gradients(image):
	print("---> Running Sobel Filter")

	return image


def apply_gaussian_filter(image):
	gaussian_filter = GaussianFilter
	print("---> Running Gaussian Filter")

	return image


def run_edge_detector(image):
	image = apply_gaussian_filter(image)
	image = compute_gradients(image)
	image = run_suppression(image)
	image = run_double_threshold(image)
	image = run_edge_tracing_with_histeresis(image)

	return image

def main():
	image = utils.load_image(image_name = "stop_sign_01.jpg")
	image = run_edge_detector(image)

	utils.show_image(image)

if __name__ == "__main__":
	main()