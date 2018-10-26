import math
import numpy as np
import math

from Task_03.base_filter import BaseFilter
import utils

class SobelFilter(BaseFilter):
	def __init__(self):
		super().__init__("sobel_filter", 3)

	def generate_filter(self):
		self.filter = np.array([
			[1, 0, -1],
			[2, 0, -2],
			[1, 0, -1]
		])

		return self.filter

	def filter_image(self, image):
		self.generate_filter()

		x_dim, y_dim, z_dim = image.shape
		new_image = np.empty((x_dim - self.size, y_dim - self.size, 3))
		angle_map = np.empty((x_dim - self.size, y_dim - self.size))

		new_pixel_map = np.empty((image.shape[0], image.shape[1]))
		new_pixel_map = 0.3 * image[:,:, 0] + 0.59 * image[:,:, 1] + 0.11 * image[:,:, 2]
	
		for x in range(x_dim - self.size):
			for y in range(y_dim - self.size):
				# sum_r_v = 0
				# sum_g_v = 0
				# sum_b_v = 0

				# sum_r_h = 0
				# sum_g_h = 0
				# sum_b_h = 0

				sum_h = 0
				sum_v = 0

				sum_h = np.sum(new_pixel_map[x:x+self.size, y:y+self.size] * self.filter)
				sum_v = np.sum(new_pixel_map[x:x+self.size, y:y+self.size] * np.transpose(self.filter))


				# for i in range(self.size):
				# 	for j in range(self.size):

				# 		new_pixel_value = 0.3 * image[x+i][y+j][0] + 0.59 * image[x+i][y+j][1] + 0.11 * image[x+i][y+j][2]
						
				# 		# sum_r_h += image[x+i][y+j][0] * self.filter[i][j]
				# 		# sum_g_h += image[x+i][y+j][1] * self.filter[i][j]
				# 		# sum_b_h += image[x+i][y+j][2] * self.filter[i][j]

				# 		# sum_r_v += image[x+i][y+j][0] * self.filter[j][i]
				# 		# sum_g_v += image[x+i][y+j][1] * self.filter[j][i]
				# 		# sum_b_v += image[x+i][y+j][2] * self.filter[j][i]

				# 		sum_h += new_pixel_value * self.filter[i][j]
				# 		sum_v += new_pixel_value * self.filter[j][i]

				# new_image[x][y][0] = math.sqrt(math.pow(sum_r_h, 2) + math.pow(sum_r_v, 2))
				# new_image[x][y][1] = math.sqrt((math.pow(sum_g_h, 2) + math.pow(sum_g_v, 2)))
				# new_image[x][y][2] = math.sqrt((math.pow(sum_b_h, 2) + math.pow(sum_b_v, 2)))

				angle_map[x][y] = math.fabs((math.atan2(sum_v, sum_h) * 180) / math.pi)
				new_image[x][y][0:3] = math.sqrt(math.pow(sum_h, 2) + math.pow(sum_v, 2))

		new_image = new_image.astype(int)
		new_image = np.clip(new_image, 0, 255)
		return new_image, angle_map