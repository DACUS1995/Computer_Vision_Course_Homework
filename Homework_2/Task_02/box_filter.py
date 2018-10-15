import math
import numpy as np

from base_filter import BaseFilter

class BoxFilter(BaseFilter):
	def __init__(self, size):
		super().__init__("box_filter", size)

	def generate_filter(self):
		self.filter = np.ones(shape=(self.size, self.size))
		self.filter /= self.filter.size
	
		return self.filter