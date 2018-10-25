import numpy as np

import utils
import Task_03.main as fn
from Task_04.template_finder import TemplateFinder

MULTI_DETECTION = True

def extract_edges(image):
	print("--> Preprocessing target image")
	image = fn.apply_gaussian_filter(image)
	image, _ = fn.compute_gradients(image)
	image, strong_edges, weak_edges = fn.run_double_threshold(image)
	image = fn.run_edge_tracing_with_histeresis(image, strong_edges, weak_edges)

	# utils.show_image(image)

	return image

def search_template(image, preprocessed_image, template_image, multi_detection = False):
	print("--> Seaching for template image")
	return TemplateFinder.find_template(image, preprocessed_image, template_image, multi_detection)

def compute_results(score_list, bounded_images):
	print("--> Computing results")
	return bounded_images[np.argmax(score_list)]

def run_image_detection(image, template_image):
	preprocessed_image = extract_edges(image)

	# score_list, bounded_images = search_template(image, preprocessed_image, template_image)
	score_list, bounded_images = search_template(image, preprocessed_image, template_image, multi_detection=MULTI_DETECTION)

	resulted_image = compute_results(score_list, bounded_images)

	return resulted_image

def main():
	image = utils.load_image(image_name = "stop_signs/stop_sign_06.jpg")
	template_image = utils.load_image(image_name = "template.png")

	resulted_image = run_image_detection(image, template_image)
	utils.show_image(resulted_image)

if __name__ == "__main__":
	main()
