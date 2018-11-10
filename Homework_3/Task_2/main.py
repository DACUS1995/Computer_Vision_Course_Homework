import numpy as np

import utils

def run_skin_hue_threshold(hsv_image, rgb_image):
	print("---> Applying skin hue threshold")
	if len(hsv_image.shape) != 3:
		raise Exception("Wrong hvs_image size")

	utils.show_image(hsv_image)

	resulted_image = np.zeros([hsv_image.shape[0], hsv_image.shape[1]])

	resulted_image[
		(
			(
				(hsv_image[:, :, 0] < 0.138) # 50
				& (hsv_image[:, :, 1] < 0.68) 
				& (hsv_image[:, :, 2] > 0.23)
			)
			&
			(
				(rgb_image[:, :, 0] > 0.37)
				& (rgb_image[:, :, 1] > 0.15)
				& (rgb_image[:, :, 2] > 0.07)
				& (rgb_image[:, :, 0] > rgb_image[:, :, 1])
				& (rgb_image[:, :, 0] > rgb_image[:, :, 2])
				& (rgb_image[:, :, 0] - rgb_image[:, :, 1] > 0.05)
			)
		)] = 1

	return resulted_image

def main():
	image = utils.load_image("5.jpg")
	hsv_image = utils.rgb_to_hsv(image)

	resulted_image = run_skin_hue_threshold(hsv_image, rgb_image=image)
	utils.show_image(resulted_image)


if __name__ == "__main__":
	main()