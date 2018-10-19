import numpy as np

import utils
import Task_03.main as fn

def extract_edges(image):
	print("---> Preprocessing target image")
	image = fn.apply_gaussian_filter(image)
	image, _ = fn.compute_gradients(image)
	image, strong_edges, weak_edges = fn.run_double_threshold(image)
	image = fn.run_edge_tracing_with_histeresis(image, strong_edges, weak_edges)
	utils.show_image(image)

	return image

def search_template(image, template_image):
	print("---> Seaching for template image")
	return image

def compute_results(image, score_map):
	print("---> Computing results")
	return image

def run_image_detection(image, template_image):
	preprocessed_image = extract_edges(image)
	score_map = search_template(preprocessed_image, template_image)
	resulted_image = compute_results(image, score_map)

	return resulted_image

def main():
	image = utils.load_image(image_name = "stop_signs/stop_sign_01.jpg")
	template_image = utils.load_image(image_name = "template.png")

	resulted_image = run_image_detection(image, template_image)
	utils.show_image(resulted_image)

if __name__ == "__main__":
	main()
