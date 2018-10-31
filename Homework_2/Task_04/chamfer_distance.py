import numpy as np
import cv2

import utils

class ChamferDistance():
	@staticmethod
	def find_template(original_image, image_map, template):
		score_list = []
		bounded_images = []
		
		# Create several scalling factors
		for scale in np.linspace(0.1, 1.2, 8):
			print(":: Using scale: ", scale)

			resized_template = ChamferDistance.resize_image(template, scale)

			# round all non-zero pixel for consistent reward for mixel matching
			resized_template = np.ceil(resized_template)

			# Check that the image is bigger than the template
			if original_image.shape[0] < resized_template.shape[0] or original_image.shape[1] < resized_template.shape[1]:
				break

			score_map = ChamferDistance.search_smallest_distance(image_map, resized_template)

			smallest_value = np.amin(score_map)
			smallest_value_pos = np.argmin(score_map)
			smallest_value_pos = ((int)(smallest_value_pos / score_map.shape[1]), (int)(smallest_value_pos % score_map.shape[1]))

			assert (smallest_value == score_map[smallest_value_pos]), "Wrong coordinates"

			bounded_image = ChamferDistance.draw_bounded_boxes(
				original_image,
				[
					(
						(smallest_value_pos[1]),
						(smallest_value_pos[0]),
						(smallest_value_pos[1] + resized_template.shape[1]),
						(smallest_value_pos[0] + resized_template.shape[0])
					)
				]
			)

			# utils.show_image(bounded_image)

			score_list.append(smallest_value)
			bounded_images.append(bounded_image)

		return score_list, bounded_images


	@staticmethod
	def search_smallest_distance(image_map, template):
		template_x_size = template.shape[0]
		template_y_size = template.shape[1]

		score_map = np.empty((image_map.shape[0] - template_x_size, image_map.shape[1] - template_y_size))

		# utils.show_image(template)

		for x in range(image_map.shape[0] - template_x_size):
			for y in range(image_map.shape[1] - template_y_size):
				sum = 0

				sum = np.sum(image_map[x:x+template_x_size, y:y+template_y_size, 0] * template[:,:,0])

				# for i in range(template_x_size):
				# 	for j in range(template_y_size):
				# 		if template[i][j][0] == 1:
				# 			sum += image_map[i+x][j+y][0]
							
				score_map[x][y] = sum / np.count_nonzero(template[:,:,0])
		utils.show_image(score_map)
		return score_map

	@staticmethod
	def compute_distance_map(image):
		new_image = ChamferDistance.init_map(image)

		# Forward pass
		for i in range(image.shape[0] - 1):
			for j in range(image.shape[1] - 1):
				temp_matrix = new_image[i:i+2, j:j+2]

				flatten_smallest_pos = np.argmin(temp_matrix)
				smallest_value = np.amin(temp_matrix)
				smallest_value_coord = ((int)(flatten_smallest_pos / image.shape[1]), (int)(flatten_smallest_pos % image.shape[1]))

				temp_matrix[temp_matrix > smallest_value] = smallest_value + 1

		# Backward pass
		for i in range(image.shape[0] - 1, 0, - 1):
			for j in range(image.shape[1] - 1, 0, -1):
				temp_matrix = new_image[i:i+2, j:j+2]

				flatten_smallest_pos = np.argmin(temp_matrix)
				smallest_value = np.amin(temp_matrix)
				smallest_value_coord = ((int)(flatten_smallest_pos / image.shape[1]), (int)(flatten_smallest_pos % image.shape[1]))

				temp_matrix[temp_matrix > smallest_value] = smallest_value + 1

		utils.show_image(new_image)
		return new_image

	@staticmethod
	def init_map(image):
		new_image = np.copy(image)

		pos_zero = new_image == 0
		pos_non_zero = new_image > 0 

		new_image[pos_zero] = 100
		new_image[pos_non_zero] = 0
		return new_image

	@staticmethod
	def resize_image(image, scale):
		resized_image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
		return resized_image


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