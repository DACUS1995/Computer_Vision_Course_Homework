import math
import numpy as np

from base_filter import BaseFilter

class GaussianFilter(BaseFilter):
	def __init__(self, size):
		super().__init__("gaussian_filter", size)

	def generate_filter(self, sigma=1):
		length = int(self.size / 2)
		# height = int(self.size / 2)

		one_d_filter = [np.exp(-z * z / (2 * sigma * sigma)) / np.sqrt(2 * np.pi * sigma * sigma) for z in range(-length, length + 1)] 
		self.filter = np.outer(one_d_filter, one_d_filter)
		self.filter /= self.filter.sum()

		return self.filter