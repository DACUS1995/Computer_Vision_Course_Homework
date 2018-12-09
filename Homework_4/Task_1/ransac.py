import numpy as np
import math
import cv2 as cv
import sys

import utils

class RANSAC():
	def __init__(self):
		raise Exception("Only static methods. Do not instantiate!")

	DISTANCE_THRESHOLD = 2
	INLINERS_THRESHOLD = 100

	@staticmethod
	def search(edge_map, component_class=1, number_of_samples=2, max_iterations=2000, output_image=None, original_image=None, total_classes_number=0) -> np.ndarray:
		print("--> Searching using class: [", component_class, "]")
		# Select just points from the specified class
		x, y = np.where(edge_map == component_class)
		points = np.column_stack((x, y))

		new_image = np.zeros(edge_map.shape) if output_image is None else output_image
		# new_image = new_image if original_image is None else original_image

		lines_eq = []
		lines_points = []
		lines_class = []

		for _ in range(max_iterations):
			print("Iteration [", _,"] Class [", component_class ,"/", total_classes_number, "] Points remaining: ", points.shape[0])
			if points.shape[0] <= 1:
				break

			sample_points = np.random.choice(points.shape[0], size=number_of_samples, replace=False)
			sample_points = np.array(points[sample_points])
			number_of_inliners, inline_points, line_eq = RANSAC.evaluate_samples(points, sample_points)
			inline_points = np.array(inline_points)

			if number_of_inliners > RANSAC.INLINERS_THRESHOLD:
				# Find the ends of the line
				# Sort the inline points by the value of the slope so that we use the correct max values
				first_point = None
				second_point = None
				if line_eq[0] > 1:
					first_point = inline_points[np.argmin(inline_points[:,1])]
					second_point = inline_points[np.argmax(inline_points[:,1])]
				else:
					first_point = inline_points[np.argmin(inline_points[:,0])]
					second_point = inline_points[np.argmax(inline_points[:,0])]

				lines_eq.append(line_eq)
				lines_points.append((first_point, second_point))
				lines_class.append(component_class)
				
				print((first_point, second_point))
				new_image = utils.draw_line(new_image, first_point, second_point)

				# Remove the points used in the last search
				for j in range(len(inline_points)):
					index = np.where((points[:, 0] == inline_points[j][0]) & (points[:, 1] == inline_points[j][1]))
					points = np.delete(points, index, axis=0)
				

		# utils.show_image(new_image)
		return new_image, lines_eq, lines_points, lines_class
		


	@staticmethod
	def evaluate_samples(points, sample_points, limit=5) -> tuple:
		counter = 0
		inline_points = []
		# Determine the equation that represents the line of the two points
		m, b, c = RANSAC.line_model(sample_points)

		for i in range(points.shape[0]):
			point = points[i]
			if m == 200:
				x0 = sample_points[0,0]
				y0 = point[1]
			else:
				x0, y0 = RANSAC.closest_point_to_line(m, b, point)
			distance = math.sqrt((x0 - point[0]) ** 2 + (y0 - point[1]) ** 2)
			
			if distance < RANSAC.DISTANCE_THRESHOLD:
				counter += 1
				inline_points.append(point)

		return counter, inline_points, (m, b, c)

	@staticmethod
	def line_model(points):
		m = (points[1, 1] - points[0, 1]) / (points[1, 0] - points[0, 0] + sys.float_info.epsilon)
		b = points[0, 1] - m * points[0, 0]
		# if points[1, 0] - points[0, 0] >= 0 and points[1, 0] - points[0, 0] <= 2: m = 200
		c = None
		if b > 5000 or b < -5000: 
			m = 200
			c = points[1, 0]
		return m, b, c

	@staticmethod
	def closest_point_to_line(m, b, point) -> tuple:
		x = (point[0] + m * point[1] - m * b) / (1 + m ** 2)
		y = (m * point[0] + (m ** 2) * point[1] - ( m ** 2) * b) / (1 + m ** 2) + b
		return x, y
