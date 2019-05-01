import numpy as np
import cv2
import sys
import glob
from scipy import signal


def sift_sim(path_a, path_b):
		'''
		Use SIFT features to measure image similarity
		@args:
			{str} path_a: the path to an image file
			{str} path_b: the path to an image file
		@returns:
			TODO
		'''
		# initialize the sift feature detector
		orb = cv2.ORB_create()

		# get the images
		img_a = cv2.imread(path_a)
		img_b = cv2.imread(path_b)

		# find the keypoints and descriptors with SIFT
		kp_a, desc_a = orb.detectAndCompute(img_a, None)
		kp_b, desc_b = orb.detectAndCompute(img_b, None)

		# initialize the bruteforce matcher
		bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

		# match.distance is a float between {0:100} - lower means more similar
		matches = bf.match(desc_a, desc_b)
		similar_regions = [i for i in matches if i.distance < 70]
		if len(matches) == 0:
				return 0
		return len(similar_regions) / len(matches)


def correlation_coefficient(im1, im2):
	if im1.shape != im2.shape:
		dim = (im2.shape[1], im2.shape[0])
		im1 = cv2.resize(im1, dsize = dim);
	product = np.mean((im1 - im1.mean()) * (im2 - im2.mean()))
	stds = im1.std() * im2.std()
	if stds == 0:
		return 0
	else:
		product /= stds
		return product

if __name__ == '__main__':

	slide_name = []
	ppt_name = []
	ppt_stor = {}

	for filename in glob.iglob('./Dataset/*/*.jpg', recursive=True):
		if filename[-7:] == 'ppt.jpg':
			ppt_name.append(filename)
			ppt_stor[filename] = cv2.imread(filename, 0)
		else:
			slide_name.append(filename)

	correct = 0
	for slide in slide_name:
		mx = -1.0
		mxname = ''
		img = cv2.imread(slide, 0)
		for ppt in ppt_name:
			# sift_val = sift_sim(slide, ppt)
			cor = correlation_coefficient( img, ppt_stor[ppt] )
			if cor > mx:
				mx = cor
				mxname = ppt
		a = slide[10:14]
		if slide[14] != '/':
			a += slide[14]
		b = mxname[10:14]
		if mxname[14] != '/':
			b += mxname[14]
		if a == b:
			correct += 1
		print(a, b, correct)
