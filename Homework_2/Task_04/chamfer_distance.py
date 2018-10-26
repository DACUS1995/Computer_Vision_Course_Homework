import numpy as np
import utils

class ChamferDistance():
	@staticmethod
	def search_smallest_distance(image_map, template):
		template_x_size = template.shape[0]
		template_y_size = template.shape[1]

		score_map = np.empty((image_map.shape[0] - template_x_size, image_map.shape[1] - template_y_size))

		utils.show_image(template)

		for x in range(image_map.shape[0] - template_x_size):
			for y in range(image_map.shape[1] - template_y_size):
				sum = 0

				sum = np.sum(image_map[x:x+template_x_size, y:y+template_y_size, 0] * template[:,:,0])

				# for i in range(template_x_size):
				# 	for j in range(template_y_size):
				# 		if template[i][j][0] == 1:
				# 			sum += image_map[i+x][j+y][0]
							
				score_map[x][y] = sum
		utils.show_image(score_map)

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
