import cv2
import numpy as np

import utils
import matplotlib.pyplot as plt

SCORE_THRESHOLD = 2000

class TemplateFinder():
	@staticmethod
	def find_template(original_image, image, template, multi_detection):
		score_list = []
		bounded_images = []

		# Create several scalling factors
		for scale in np.linspace(0.1, 1.2, 8):
			print(":: Using scale: ", scale)
			resized_template = TemplateFinder.resize_image(template, scale)

			# round all non-zero pixel for consistent reward for mixel matching
			resized_template = np.ceil(resized_template)

			# Check that the image is bigger than the template
			if image.shape[0] < resized_template.shape[0] or image.shape[1] < resized_template.shape[1]:
				break

			score_map = TemplateFinder.conv_search(image, resized_template)

			#TODO add multi detection + try Chamfer distance
			if multi_detection == False:
				largest_value_index = np.argmax(score_map)
				corner_upper_left = ((int)(largest_value_index / score_map.shape[1]), (int)(largest_value_index % score_map.shape[1]))

				new_image = TemplateFinder.draw_bounded_boxes(
					original_image,
					[
						(
							(corner_upper_left[1]),
							(corner_upper_left[0]),
							(corner_upper_left[1] + resized_template.shape[1]),
							(corner_upper_left[0] + resized_template.shape[0])
						)
					]
				)

				score_list.append(score_map[corner_upper_left[0]][corner_upper_left[1]])
				bounded_images.append(new_image)

				# --- Show additional info ---
				print(corner_upper_left) # position of the highest value
				print(score_map[corner_upper_left[0]][corner_upper_left[1]]) # the highest value
				print(np.amax(score_map))
				utils.show_image(new_image)
				utils.show_image(score_map)
				# utils.show_image(resized_template)

			else:
				# I used as threshold a score of 2000
				highest_scores_indices = np.where(score_map > SCORE_THRESHOLD)
				highest_scores_indices = [
					(
						highest_scores_indices[1][i], 
						highest_scores_indices[0][i],
						highest_scores_indices[1][i] + resized_template.shape[1], 
						highest_scores_indices[0][i] + resized_template.shape[0],
					) for i in range(highest_scores_indices[0].size)
				]

				new_image = TemplateFinder.draw_bounded_boxes(original_image, highest_scores_indices)
				score_list.append(np.amax(score_map))
				bounded_images.append(new_image)

				# --- Show additional info ---
				print(np.amax(score_map))
				utils.show_image(new_image)
				utils.show_image(score_map)
	
		return (score_list, bounded_images)
		
		
	@staticmethod
	def resize_image(image, scale):
		resized_image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
		return resized_image

	@staticmethod
	def conv_search(image, template):
		template_x_size = template.shape[0]
		template_y_size = template.shape[1]

		# print(np.amax(image))
		# print(np.amax(template))

		score_map = np.empty((image.shape[0] - template_x_size, image.shape[1] - template_y_size))

		#Make sure the image tensor is range bounded to 0-1 values
		image = image / np.amax(image)

		for x in range(image.shape[0] - template_x_size):
			for y in range(image.shape[1] - template_y_size):
				score_map[x][y] = np.sum(image[x : x + template_x_size, y : y + template_y_size, 0] * template[:,:,0])
				# It may be better to not use the score normalization based on the number of active pixels in the template image
				# score_map[x, y] = (score_map[x, y]) / (np.where(template > 0)[0].size)
		return score_map

	@staticmethod
	def draw_bounded_boxes(image, rect_coords):
		new_image = np.copy(image)

		for index in range(len(rect_coords)):
			cv2.rectangle(
				new_image, 
				(
					(rect_coords[index][0]),
					(rect_coords[index][1])
				),
				(
					(rect_coords[index][2]),
					(rect_coords[index][3])
				),
				(0, 0, 255),
				2
			)

		return new_image
