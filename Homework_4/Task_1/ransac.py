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
	def search(edge_map, component_class=1, number_of_samples=2, max_iterations=2000):
		print("--> Searching using class: [", component_class, "]")
		# Select just points from the specified class
		x, y = np.where(edge_map == component_class)
		points = np.column_stack((x, y))

		new_image = np.zeros(edge_map.shape)
		for _ in range(max_iterations):
			print("Iteration [", _,"]  Points remaining: ", points.shape[0])
			if points.shape[0] == 0:
				break

			sample_points = np.random.choice(points.shape[0], size=number_of_samples, replace=False)
			sample_points = np.array(points[sample_points])
			number_of_inliners, inline_points, line_eq = RANSAC.evaluate_samples(points, sample_points)

			if number_of_inliners > RANSAC.INLINERS_THRESHOLD:
				# Find the ends of the line
				# Sort the inline points by the value of the slope so that we use the correct max values
				if line_eq[0] > 1:
					inline_points.sort(key=lambda x : x[0])
					print(line_eq, " ", inline_points[0], " ", inline_points[len(inline_points) - 1], " ", len(inline_points))

				else:
					inline_points.sort(key=lambda x : x[1])
					print(line_eq, " ", inline_points[0], " ", inline_points[len(inline_points) - 1], " ", len(inline_points))

				new_image = utils.draw_line(new_image, inline_points[0], inline_points[len(inline_points) - 1])

				# Remove the points used in the last search
				for j in range(len(inline_points)):
					index = np.where((points[:, 0] == inline_points[j][0]) & (points[:, 1] == inline_points[j][1]))
					points = np.delete(points, index, axis=0)
				
				# l = 0
				# end = points.shape[0]
				# while l <= end:
				# 	for j in range(len(inline_points)):
				# 		if(l >= end): break
				# 		print(end)
				# 		if np.array_equal(points[l], inline_points[j]):
				# 			points = np.delete(points, l, 0)
				# 			end -= 1
				# 	l += 1

				# image = utils.draw_line(edge_map, inline_points[0], inline_points[len(inline_points) - 1])
				# image = utils.draw_line_eq(edge_map, line_eq)
				
				# utils.show_image(new_image)
				# print(i, ": ", number_of_inliners)

		utils.show_image(new_image)
		


	@staticmethod
	def evaluate_samples(points, sample_points, limit=5) -> tuple:
		counter = 0
		inline_points = []
		# Determine the equation that represents the line of the two points
		m, b = RANSAC.line_model(sample_points)

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

		return counter, inline_points, (m, b)

	@staticmethod
	def line_model(points):
		m = (points[1, 1] - points[0, 1]) / (points[1, 0] - points[0, 0] + sys.float_info.epsilon)
		b = points[0, 1] - m * points[0, 0]
		if points[1, 0] - points[0, 0] >= 0 and points[1, 0] - points[0, 0] <= 2: m = 200
		return m, b

	@staticmethod
	def closest_point_to_line(m, b, point) -> tuple:
		x = (point[0] + m * point[1] - m * b) / (1 + m ** 2)
		y = (m * point[0] + (m ** 2) * point[1] - ( m ** 2) * b) / (1 + m ** 2) + b
		return x, y
