import math
import numpy as np

from ..Task_02.base_filter import BaseFilter

class SobelFilter(BaseFilter):
    __init__(self):
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

        for x in range(x_dim - self.size):
			for y in range(y_dim - self.size):
				sum_r = 0
				sum_g = 0
				sum_b = 0

				for i in range(self.size):
					for j in range(self.size):
						sum_r += image[x+i][y+j][0] * self.filter[i][j]
						sum_g += image[x+i][y+j][1] * self.filter[i][j]
						sum_b += image[x+i][y+j][2] * self.filter[i][j]

				new_image[x][y][0] = sum_r
				new_image[x][y][1] = sum_g
				new_image[x][y][2] = sum_b

		new_image = new_image.astype(int)
		return new_image