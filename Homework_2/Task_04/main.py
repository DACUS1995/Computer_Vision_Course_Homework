import numpy as np

import utils
import Task_03.main as fn
from Task_04.template_finder import TemplateFinder
from Task_04.chamfer_distance import ChamferDistance

MULTI_DETECTION = False

def extract_edges(image):
	print("--> Preprocessing target image")
	image = fn.apply_gaussian_filter(image)
	image, _ = fn.compute_gradients(image)
	image, strong_edges, weak_edges = fn.run_double_threshold(image)
	image = fn.run_edge_tracing_with_histeresis(image, strong_edges, weak_edges)

	# utils.show_image(image)

	return image

def search_using_chamfer_distance(original_image, preprocessed_image, template):
	print("--> Running Chamfer Distance")
	image_map = ChamferDistance.compute_distance_map(preprocessed_image)
	return ChamferDistance.find_template(original_image, image_map, template)

def search_template(image, preprocessed_image, template_image, multi_detection = False):
	print("--> Seaching for template image")
	return TemplateFinder.find_template(image, preprocessed_image, template_image, multi_detection)

def compute_results(score_list, bounded_images, method_used):
	print("--> Computing results")
	if method_used == "template_matching":
		return bounded_images[np.argmax(score_list)]
	else:
		return bounded_images[np.argmin(score_list)]

def run_image_detection(image, template_image):
	preprocessed_image = extract_edges(image)

	score_list, bounded_images = search_using_chamfer_distance(image, preprocessed_image, template_image)

	# score_list, bounded_images = search_template(image, preprocessed_image, template_image)
	# score_list, bounded_images = search_template(image, preprocessed_image, template_image, multi_detection=MULTI_DETECTION)

	resulted_image = compute_results(score_list, bounded_images, method_used="template_matching")

	return resulted_image

def main():
	image = utils.load_image(image_name = "stop_signs/stop_sign_01.jpg")
	template_image = utils.load_image(image_name = "template.png")

	resulted_image = run_image_detection(image, template_image)
	utils.show_image(resulted_image)

if __name__ == "__main__":
	main()
