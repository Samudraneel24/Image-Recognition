import numpy as np
import cv2
import sys
import glob

def correlation_coefficient(im1, im2):
	'''Calculate normalised cross correlation between im1 and im2'''
	if im1.shape != im2.shape:
		dim = (im2.shape[1], im2.shape[0])
		im1 = cv2.resize(im1, dsize = dim)

	im1 = cv2.adaptiveThreshold(im1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
	im2 = cv2.adaptiveThreshold(im2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)

	product = np.mean((im1 - im1.mean()) * (im2 - im2.mean()))
	stds = im1.std() * im2.std()
	if stds == 0:
		return 0
	else:
		product /= stds
		return product

if __name__ == '__main__':

	file1 = open('20171065_20171185_20171022.txt', 'w')
	slide_name = []
	ppt_name = []
	ppt_stor = {}

	ppt_folder = sys.argv[1]
	slide_folder = sys.argv[2]
	for filename in glob.iglob(slide_folder + '/*.jpg', recursive=True):
		slide_name.append(filename)

	for filename in glob.iglob(ppt_folder + '/*.jpg', recursive=True):
		ppt_name.append(filename)
		ppt_stor[filename] = cv2.imread(filename, 0)

	for slide in slide_name:
		mx = -1.0
		mxname = ''
		img = cv2.imread(slide, 0)
		for ppt in ppt_name:
			cor = correlation_coefficient( img, ppt_stor[ppt] )
			if cor > mx:
				mx = cor
				mxname = ppt

		# image paths extraction
		slide_op = slide[len(slide_folder)+1 : len(slide)]
		ppt_op = mxname[len(ppt_folder)+1 : len(mxname)]
		print(slide_op, ppt_op, file = file1)
	file1.close()
