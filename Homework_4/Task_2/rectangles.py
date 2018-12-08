import numpy as np
import utils

from typing import Dict, Tuple, List

class Rectangle_finder:
	def __init__(self):
		pass

	@staticmethod
	def search(lines_eq, lines_points, lines_classes) -> list:
		
		# First we go through the lines that belong to the same connected component
		class_set = set(lines_classes)

		lines_classes = np.array(lines_classes)
		lines_eq = np.array(lines_eq)
		lines_points = np.array(lines_points)
		rectangles = []

		for class_ID in class_set:
			line_indices = np.where(lines_classes == class_ID)

			current_lines_eq = lines_eq[line_indices]
			current_lines_points = lines_points[line_indices]

			slope_dict = {}
			for i in range(current_lines_eq.shape[0]):
				current_slope = round(current_lines_eq[i, 0], 1)
				if current_slope in slope_dict:
					slope_dict[current_slope].append(current_lines_eq[i])
				else:
					slope_dict[current_slope] = [current_lines_eq[i]]
			print("class: ", class_ID)
			print(slope_dict)
			print("-------------------")

			for slope_one, lines_eq_arr_one in slope_dict.items():
				for slope_two, lines_eq_arr_two in slope_dict.items():
					if slope_one == slope_two: continue
					if len(lines_eq_arr_one) < 2: continue
					if len(lines_eq_arr_two) < 2: continue
					
					for i in range(len(lines_eq_arr_one) - 1):
						for j in range(i, len(lines_eq_arr_one)):

							for l in range(len(lines_eq_arr_two) - 1):
								for k in range(l, len(lines_eq_arr_two)):

									if i ==j: continue
									if l == k: continue

									m1, b1, c1 = lines_eq_arr_one[i]
									m2, b2, c2 = lines_eq_arr_one[j]

									m3, b3, c3 = lines_eq_arr_two[l]
									m4, b4, c4 = lines_eq_arr_two[k]

									# Vertical lines handling
									if m1 == 200:
										x1 = c1
										x2 = c1
										b1 = 0
										y1 = m3 * x1 + b3
										y2 = m4 * x2 + b4
									else:
										x1 = (b3 - b1) / (m1 - m3)
										x2 = (b4 - b1) / (m1 - m4)
										y1 = m3 * x1 + b3
										y2 = m4 * x2 + b4

									if m2 == 200:
										x3 = c2
										x4 = c2
										b2 = 0
										y3 = m3 * x3 + b3
										y4 = m4 * x4 + b4
									else:
										x3 = (b3 - b2) / (m2 - m3)
										x4 = (b4 - b2) / (m2 - m4)
										y3 = m3 * x3 + b3
										y4 = m4 * x4 + b4

									if m3 == 200:
										x1 = c3
										x3 = c3
										b3 = 0
										y1 = m1 * x1 + b1
										y3 = m2 * x3 + b2
									
									if m4 == 200:
										x2 = c4
										x4 = c4
										b4 = 0
										y2 = m1 * x2 + b1
										y4 = m2 * x4 + b2


									rectangles.append(((y1, x1), (y2, x2), (y3, x3), (y4, x4)))
									
		return rectangles

if __name__ == "__main__":
	raise Exception("Must be used only as a module")