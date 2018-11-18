import numpy as np
import math
import cv2 as cv

import utils

class RANSAC():
	def __init__(self):
		raise Exception("Only static methods. Do not instantiate!")

	DISTANCE_THRESHOLD = 10
	INLINERS_THRESHOLD = 50

	@staticmethod
	def search(edge_map, component_class=1, number_of_samples=2, max_iterations=50):
		# Select just points from the specified class
		x, y = np.where(edge_map == component_class)
		points = np.column_stack((x, y))

		for i in range(max_iterations):
			sample_points = np.random.choice(points.shape[0], size=number_of_samples, replace=False)
			sample_points = np.array(points[sample_points])
			number_of_inliners, inliner_points, line_eq = RANSAC.evaluate_samples(points, sample_points)

			if number_of_inliners > RANSAC.INLINERS_THRESHOLD:
				# image = utils.draw_line(edge_map, sample_points[0], sample_points[1])
				image = utils.draw_line_eq(edge_map, line_eq)
				
				utils.show_image(image)
				print(i, ": ", number_of_inliners)


	@staticmethod
	def evaluate_samples(points, sample_points, limit=5):
		counter = 0
		inliner_points = []
		# Determine the equation that represents the line of the two points
		m, b = RANSAC.line_model(sample_points)

		for i in range(points.shape[0]):
			point = points[i]
			x0, y0 = RANSAC.closest_point_to_line(m, b, point)
			distance = math.sqrt((x0 - point[0]) ** 2 + (y0 - point[1]) ** 2)
			
			if distance < RANSAC.DISTANCE_THRESHOLD:
				counter += 1
				inliner_points.append(point)

		return counter, inliner_points, (m, b)

	@staticmethod
	def line_model(points):
		m = (points[1, 1] - points[0, 1]) / (points[1, 0] - points[0, 0])
		b = points[0, 1] - m * points[0, 0]
		return m, b

	@staticmethod
	def closest_point_to_line(m, b, point):
		x = (point[0] + m * point[1] - m * b) / (1 + m ** 2)
		y = (m * point[0] + (m ** 2) * point[1] - ( m ** 2) * b) / (1 + m ** 2) + b
		return x, y
