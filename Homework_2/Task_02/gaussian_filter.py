import math
import numpy as np

from base_filter import BaseFilter

class GaussianFilter(BaseFilter):
	def __init__(self, size):
		super().__init__("gaussian_filter", size)

	def generate_filter(self, sigma=1.5):
		sum = 0

		length = int(self.size / 2)
		height = int(self.size / 2)

		# x, y = np.ogrid[-length:length, -height:height]
		# h = np.exp( -(np.power(x, 2) + np.power(y, 2)) / (2. * sigma*sigma) )
		# h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
		# sumh = h.sum()
		# if sumh != 0:
		#     h /= sumh
		# return h

		# for x in range(-length, length + 1):
		# 	for y in range(-height, height + 1):
		# 		self.filter[x][y] = np.exp(-(x * x + y * y) / (2 * sigma * sigma)) / (2 * np.pi * sigma * sigma)

		s, k = 1, int(self.size / 2) #  generate a (2k+1)x(2k+1) gaussian kernel with mean=0 and sigma = s
		probs = [np.exp(-z * z / (2 * s * s)) / np.sqrt(2 * np.pi * s * s) for z in range(-k, k + 1)] 
		self.filter = np.outer(probs, probs)	

		return self.filter